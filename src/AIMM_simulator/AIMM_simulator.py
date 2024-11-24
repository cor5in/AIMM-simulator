# Geon Kim 2024-11-12 Version
# Simulation structure:
# Sim - Scenario - MME
# |
# RIC
# |
# Cell---Cell---Cell--- ....
# |       |      |
# UE UE  UE UE  UE UE ...

from .UMa_pathloss_model import UMa_pathloss
from collections import deque
from os.path import basename
__version__ = '2.0.3'
'''The AIMM simulator emulates a cellular radio system roughly following 5G concepts and channel models.'''

from sys import stderr, stdout, exit, version as pyversion
from math import hypot, atan2, pi as math_pi
from time import time, sleep
try:
  import numpy as np
except:
  print('numpy not found: please do "pip install numpy"', file=stderr)
  exit(1)
try:
  import simpy
except:
  print('simpy not found: please do "pip install simpy"', file=stderr)
  exit(1)
from .NR_5G_standard_functions import SINR_to_CQI, CQI_to_64QAM_efficiency


def np_array_to_str(x):
  ' Formats a 1-axis np.array as a tab-separated string '
  return np.array2string(x, separator='\t').replace('[', '').replace(']', '')


def _nearest_weighted_point(x, pts, w=1.0):
  '''
  Internal use only.
  Given a point x of shape (dim,), where dim is typically 2 or 3,
  an array of points pts of shape (npts,dim),
  and a vector of weights w of the same length as pts,
  return the index of the point minimizing w[i]*d[i],
  where d[i] is the distance from x to point i.
  Returns the index of the point minimizing w[i]*d[i].
  For the application to cellular radio systems, we let pts be the
  cell locations, and then if we set
  w[i]=p[i]**(-1/alpha),
  where p[i] is the transmit power of cell i, and alpha>=2 is the pathloss
  exponent, then this algorithm will give us the index of the cell providing
  largest received power at the point x.
  '''
  weighted_distances = w * np.linalg.norm(pts - x, axis=1)
  imin = np.argmin(weighted_distances)
  if 0:  # dbg
    print('x=', x)
    print('pts=', pts)
    print('weighted_distances=', weighted_distances)
  return weighted_distances[imin], imin


def to_dB(x):
  return 10.0 * np.log10(x)


def from_dB(x):
  return np.power(10.0, x / 10.0)


class Cell:
 '''
 Class representing a single Cell (gNB). As instances are created, they are automatically given indices starting from 0.
 This index is available as the data member ``cell.i``. The variable ``Cell.i`` is always the current number of cells.

 Parameters
 ----------
 sim : Sim
   Simulator instance which will manage this Cell.
 interval : float
   Time interval between Cell updates (default 15 minutes = 900 seconds).
 xyz : [float, float, float]
   Position of cell in metres, and antenna height.
 h_BS : float
   Antenna height in metres; only used if xyz is not provided.
 MIMO_gain_dB : float
   Effective power gain from MIMO in dB. A typical value might be 3dB for 2x2 MIMO.
 pattern : array or function
   If an array, then a 360-element array giving the antenna gain in dB in 1-degree increments (0=east, then counterclockwise).
   Otherwise, a function giving the antenna gain in dB in the direction theta=(180/pi)*atan2(y,x).
 f_callback :
   A function with signature ``f_callback(self,kwargs)``, which will be called at each iteration of the main loop.
 verbosity : int
   Level of debugging output (0=none).
 '''
 i = 0

 def __init__(self,
              sim,
              interval=15 * 60,  # 15분을 초단위로 변환
              xyz=None,
              h_BS=20.0,
              MIMO_gain_dB=0.0,
              pattern=None,
              f_callback=None,
              f_callback_kwargs={},
              verbosity=0):

   self.i = Cell.i; Cell.i += 1
   self.sim = sim
   self.interval = interval  # 15분 간격

   # 에너지 소비 추적을 위한 설정
   self.energy_per_interval = 25  # Wh (100Wh/4intervals = 25Wh per 15min)

   # 주파수 대역별 설정
   self.freq_config = {  
            800: {
                'bandwidth': 10.0,
                'n_RBs': 50, 
                'active': False,
                'energy_consumed': 0.0
            },
            1800: {
                'bandwidth': 20.0, 
                'n_RBs': 100,
                'active': False,
                'energy_consumed': 0.0
            },
            3600: {
                'bandwidth': 100.0,
                'n_RBs': 500,
                'active': False,
                'energy_consumed': 0.0
            }
        }

   # 초기 활성 주파수와 총 에너지 소비
   self.active_freqs = set()
   self.freq_users = {freq: set() for freq in self.freq_config} # 주파수별 연결된 UE
   self.total_energy_consumed = 0  # 총 에너지 소비 (Wh)
   self.intervals_count = 0  # 15분 간격 카운트

   # 주파수별 RB 수 초기화
   self.rb_masks = {
       freq: np.ones(config['n_RBs'])
       for freq, config in self.freq_config.items()
   }

   self.base_power_W = 130.0  # 상시 전력 (W)
   self.frequency_power_W = 100.0  # 주파수당 추가 전력 (W)
   self.total_energy_consumed = 0.0  # 총 에너지 소비량 (Wh)

   self.rbs = simpy.Resource(self.sim.env, capacity=50)
   self.pattern = pattern
   self.f_callback = f_callback
   self.f_callback_kwargs = f_callback_kwargs
   self.MIMO_gain_dB = MIMO_gain_dB
   self.attached = set()

   # 주파수별 리포트 관리
   self.reports = {
    freq: {
        'cqi': {},
        'rsrp': {},
        'throughput_Mbps': {}
    }
    for freq in self.freq_config.keys()
   }

   self.rsrp_history = {freq: {} for freq in self.freq_config.keys()}

   if xyz is not None:
     self.xyz = np.array(xyz)
   else:
     self.xyz = np.empty(3)
     self.xyz[:2] = 100.0 + 900.0 * self.rng.random(2)
     self.xyz[2] = h_BS

   if verbosity > 1: print(f'Cell[{self.i}] is at', self.xyz, file=stderr)
   self.verbosity = verbosity
   self.sim._set_hetnet()

   # 이웃 셀 관리를 위한 속성 추가
   self.neighbor_cells = []
   self.sleep_thresholds = {800: 0.2, 1800: 0.3, 3600: 0.4}  # 주파수별 sleep 임계값
   self.traffic_history = {freq: [] for freq in self.freq_config}  # 트래픽 이력 관리

 def get_neighbor_cells(self):
        """이웃 셀 목록 반환"""
        if not self.neighbor_cells:
            # 반경 1km 내의 셀들을 이웃으로 간주
            for cell in self.sim.cells:
                if cell.i != self.i:
                    distance = np.linalg.norm(cell.xyz[:2] - self.xyz[:2])
                    if distance <= 1000:  # 1km
                        self.neighbor_cells.append(cell)
        return self.neighbor_cells

 def predict_traffic_surge(self, freq, window_size=4):
        """단기 트래픽 증가 예측"""
        if len(self.traffic_history[freq]) < window_size:
            return False
            
        recent_loads = self.traffic_history[freq][-window_size:]
        trend = np.polyfit(range(window_size), recent_loads, 1)[0]
        return trend > 0.1  # 양의 기울기가 0.1 이상이면 증가 추세로 판단   

 def can_shift_traffic(self, freq):
        """트래픽 이동 가능 여부 확인"""
        users = self.get_frequency_users(freq)
        if not users:
            return True
            
        # 다른 활성 주파수가 있는지 확인
        alternative_freqs = [f for f in self.active_freqs if f != freq]
        if not alternative_freqs:
            return False
            
        # 각 사용자별로 다른 주파수로 이동 가능한지 확인
        for user in users:
            can_move = False
            for alt_freq in alternative_freqs:
                if self.can_satisfy_qos(alt_freq, self.get_frequency_load(alt_freq), user.qos_requirement):
                    can_move = True
                    break
            if not can_move:
                return False
                
        return True

 def redistribute_traffic(self, freq):
        """트래픽 재분배"""
        users = list(self.get_frequency_users(freq))
        alternative_freqs = [f for f in self.active_freqs if f != freq]
        
        for user in users:
            best_freq = None
            best_score = -float('inf')
            
            for alt_freq in alternative_freqs:
                load = self.get_frequency_load(alt_freq)
                if self.can_satisfy_qos(alt_freq, load, user.qos_requirement):
                    score = self.calculate_handover_score(
                        user.measure_rsrp(self, alt_freq),
                        load,
                        alt_freq
                    )
                    if score > best_score:
                        best_score = score
                        best_freq = alt_freq
                        
            if best_freq:
                self.freq_users[freq].remove(user)
                self.freq_users[best_freq].add(user)
                user.update_frequency(best_freq)

 def get_sleep_threshold(self, freq):
        """주파수별 sleep 임계값 반환"""
        return self.sleep_thresholds.get(freq, 0.3)  # 기본값 0.3

 def calculate_avg_throughput(self, freq):
        """주파수별 평균 처리량 계산"""
        users = self.get_frequency_users(freq)
        if not users:
            return 0
            
        total_throughput = sum(self.get_UE_throughput(user.i, freq) for user in users)
        return total_throughput / len(users)         

 def update_energy_consumption(self):
    """15분 간격으로 에너지 소비 업데이트"""
    # 시간 단위 변환 확인 필요
    active_count = len(self.active_freqs)
    
    # 기본 전력 + 주파수당 추가 전력 고려
    total_power = self.base_power_W + (self.frequency_power_W * active_count)
    energy_this_interval = (total_power * self.interval) / 3600  # Wh 단위로 변환
    
    self.total_energy_consumed += energy_this_interval
    self.intervals_count += 1
    
    # 주파수별 에너지 소비 분배
    if active_count > 0:
        energy_per_freq = energy_this_interval / active_count
        for freq in self.active_freqs:
            self.freq_config[freq]['energy_consumed'] += energy_per_freq

 def get_hourly_energy_consumption(self):
   """시간당 에너지 소비량 계산"""
   hourly_rate = len(self.active_freqs) * 100  # 시간당 100Wh * 활성 대역 수
   return hourly_rate

 

 def get_energy_stats(self):
   """에너지 소비 통계 반환"""
   hours_elapsed = self.intervals_count / 4  # 15분 간격을 시간으로 변환
   return {
       'total_energy_consumed': self.total_energy_consumed,  # 총 에너지 소비 (Wh)
       'hours_elapsed': hours_elapsed,
       'active_bands': len(self.active_freqs),
       'hourly_consumption_rate': self.get_hourly_energy_consumption(),
       'per_frequency': {
           freq: {
               'active': self.freq_config[freq]['active'],
               'energy_consumed': self.freq_config[freq]['energy_consumed']  # Wh
           }
           for freq in self.freq_config
       }
   }

  # Cell 클래스에 새로운 메서드 추가
  def get_frequency_priority(self, service_type):
    """
    서비스 타입에 따른 주파수 우선순위 반환
    """
    # 음성 통화인 경우 커버리지 우선
    if service_type == "voice":
        return [800, 1800, 3600] 
    
    # 데이터 서비스인 경우 대역폭/용량 우선  
    elif service_type == "data":
        # 커버리지를 고려한 가중치 적용
        weights = {
            800: 1.0,   # 기본 커버리지
            1800: 0.8,  # 중간 커버리지  
            3600: 0.6   # 제한된 커버리지
        }
        
        # 각 주파수별 실제 사용 가능한 용량 계산
        available_capacity = {}
        for freq in self.freq_config:
            bandwidth = self.freq_config[freq]['bandwidth']
            coverage_weight = weights[freq]
            available_capacity[freq] = bandwidth * coverage_weight
            
        # 용량 기준 내림차순 정렬
        return sorted(available_capacity, key=available_capacity.get, reverse=True)




 def activate_frequency(self, freq):
   """주파수 대역 활성화"""
   if freq in self.freq_config and not self.freq_config[freq]['active']:
       self.freq_config[freq]['active'] = True
       self.active_freqs.add(freq)
       return True
   return False

 def deactivate_frequency(self, freq):
   """주파수 대역 비활성화"""
   if freq in self.freq_config and self.freq_config[freq]['active']:
       self.freq_config[freq]['active'] = False
       self.active_freqs.remove(freq)
       return True
   return False

 def set_frequency(self, freq):
   """활성 주파수 변경"""
   if freq in self.freq_config and self.freq_config[freq]['active']:
       self.active_freq = freq
       return True
   return False

 def get_active_frequencies(self):
   """활성화된 주파수 대역 목록 반환"""
   return list(self.active_freqs)

 def set_rb_mask(self, mask, freq=None):
   """특정 주파수의 RB 마스크 설정"""
   if freq is None:
       freq = min(self.active_freqs) if self.active_freqs else list(self.freq_config.keys())[0]
   if freq in self.freq_config:
       if len(mask) == self.freq_config[freq]['n_RBs']:
           self.rb_masks[freq] = np.array(mask)
           return True
   return False

 def get_rb_mask(self, freq=None):
   """특정 주파수의 RB 마스크 반환"""
   if freq is None:
       freq = min(self.active_freqs) if self.active_freqs else list(self.freq_config.keys())[0]
   return self.rb_masks.get(freq)

 def get_rsrp(self, i, freq=None):
   """주파수별 RSRP 반환"""
   if freq is None:
       freq = min(s.active_freqs) if s.active_freqs else list(self.freq_config.keys())[0]
   if not self.freq_config[freq]['active']:
       return -np.inf
   if i in self.reports[freq]['rsrp']:
       rsrp = self.reports[freq]['rsrp'][i][1]
       freq_factor = (freq/800.0)**2
       return rsrp - 10*np.log10(freq_factor)
   return -np.inf

 def get_UE_throughput(self, ue_i, freq=None):  
   """주파수별 UE 처리량 반환"""
   if freq is None:
       freq = min(self.active_freqs) if self.active_freqs else list(self.freq_config.keys())[0]
   reports = self.reports[freq]['throughput_Mbps']
   if ue_i in reports:
       # RB 기반 처리량 계산
       n_rbs = self.freq_config[freq]['n_RBs']
       rb_mask = self.rb_masks[freq]
       return reports[ue_i][1] * (rb_mask.sum() / n_rbs)
   return -np.inf

 def get_UE_CQI(self, ue_i, freq=None):
   """주파수별 CQI 반환"""
   if freq is None:
       freq = min(self.active_freqs) if self.active_freqs else list(self.freq_config.keys())[0]
   reports = self.reports[freq]['cqi']
   return reports[ue_i][1] if ue_i in reports else np.nan*np.ones(self.freq_config[freq]['n_RBs'])

 def set_pattern(self,pattern):
   """Set the antenna radiation pattern."""
   self.pattern = pattern

 def set_MIMO_gain(self,MIMO_gain_dB): 
   """Set the MIMO gain in dB."""
   self.MIMO_gain_dB = MIMO_gain_dB

 def get_xyz(self):
   """Return cell position."""
   return self.xyz

 def set_xyz(self,xyz):
   """Set cell position."""
   self.xyz = np.array(xyz)
   self.sim.cell_locations[self.i] = self.xyz
   print(f'Cell[{self.i}] is now at {self.xyz}',file=stderr)

 def get_nattached(self):
   """Return number of attached UEs."""
   return len(self.attached)

 def loop(self):
    while True:
        num_active_freqs = len(self.active_freqs)
        total_power_W = self.base_power_W + self.frequency_power_W * num_active_freqs
        energy_consumed = total_power_W * (self.interval / 3600)
        self.total_energy_consumed += energy_consumed

        # Callback
        if self.f_callback is not None:
            self.f_callback(self, **self.f_callback_kwargs)
        yield self.sim.env.timeout(self.interval)

 def monitor_rbs(self):
   while True:
     if self.rbs.queue:
       if self.verbosity>0: print(f'rbs at {self.sim.env.now:.2f} ={self.rbs.count}')
     yield self.sim.env.timeout(5.0)

 def __repr__(self):
   return f'Cell(index={self.i},xyz={self.xyz})'

 def select_frequency_by_qos(self, ue, service_type, qos_requirement):
    """
    QoS 요구사항을 고려한 주파수 선택
    """
    priorities = self.get_frequency_priority(service_type)
    
    for freq in priorities:
        if not self.freq_config[freq]['active']:
            continue
            
        # 해당 주파수 대역의 현재 로드 확인
        current_load = self.get_frequency_load(freq)
        
        # QoS 만족 가능 여부 체크
        if self.can_satisfy_qos(freq, current_load, qos_requirement):
            return freq
            
    return None  # 적절한 주파수를 찾지 못한 경우

  def can_satisfy_qos(self, freq, current_load, qos_requirement):
    """주파수별 QoS 만족 여부 확인"""
    available_bandwidth = self.freq_config[freq]['bandwidth'] * (1 - current_load)
    required_bandwidth = qos_requirement.get('min_bandwidth', 0)
    return available_bandwidth >= required_bandwidth

  def get_frequency_load(self, freq):
    """주파수별 현재 RB 사용률 계산"""
    # freq가 유효한지 검사 추가
    if freq not in self.freq_config:
        raise ValueError(f"Invalid frequency: {freq}")
        
    if not self.freq_config[freq]['active']:
        return 1.0
        
    total_rbs = self.freq_config[freq]['n_RBs']
    used_rbs = sum(1 for rb in self.rb_masks[freq] if rb == 1)
    return used_rbs / total_rbs

  def get_frequency_users(self, freq):
      """주파수별 연결된 UE 목록"""
      return self.freq_users.get(freq, set())

  def evaluate_handover_target(self, ue):
    """
    다중 주파수 환경에서의 핸드오버 타겟 평가
    """
    candidate_cells = []
    
    for neighbor_cell in self.get_neighbor_cells():
        for freq in neighbor_cell.active_freqs:
            rsrp = ue.measure_rsrp(neighbor_cell, freq)
            load = neighbor_cell.get_frequency_load(freq)
            
            score = self.calculate_handover_score(rsrp, load, freq)
            candidate_cells.append({
                'cell': neighbor_cell,
                'frequency': freq, 
                'score': score
            })
    
    # 점수 기준 정렬
    return sorted(candidate_cells, key=lambda x: x['score'], reverse=True)  

  def calculate_handover_score(self, rsrp, load, freq):
    """핸드오버 후보 평가 점수 계산"""
    coverage_weight = {800: 1.0, 1800: 0.8, 3600: 0.6}
    return rsrp * coverage_weight[freq] * (1 - load)

  def generate_frequency_metrics(self):
        """주파수별 성능 지표 생성"""
        metrics = {}
        for freq in self.freq_config:
            if not self.freq_config[freq]['active']:
                continue
                
            metrics[freq] = {
                'capacity_utilization': self.get_frequency_load(freq),
                'connected_users': len(self.get_frequency_users(freq)),
                'average_throughput': self.calculate_avg_throughput(freq),
                'energy_efficiency': self.calculate_energy_efficiency(freq)
            }
        
        return metrics

  def calculate_energy_efficiency(self, freq):
    """주파수별 에너지 효율성 계산"""
    if not self.freq_config[freq]['active']:
        return 0
    
    total_throughput = self.calculate_avg_throughput(freq) * len(self.freq_users[freq])
    energy_consumed = self.freq_config[freq]['energy_consumed']
    return total_throughput / energy_consumed if energy_consumed > 0 else 0

  def evaluate_sleep_condition(self, freq):
    """주파수별 sleep 조건 평가"""
    traffic_load = self.get_frequency_load(freq)
    threshold = self.get_sleep_threshold(freq)
    
    if traffic_load < threshold:
        # 단기 트래픽 예측 확인
        if not self.predict_traffic_surge(freq):
            return True
    return False

  def activate_sleep_mode(self, freq):
    """주파수별 sleep 모드 활성화"""
    if self.can_shift_traffic(freq):
        self.deactivate_frequency(freq)
        self.redistribute_traffic(freq)
        return True
    return False  

# END class Cell

class UE:
  '''
    Represents a single UE. As instances are created, the are automatically given indices starting from 0.  This index is available as the data member ``ue.i``.   The static (class-level) variable ``UE.i`` is always the current number of UEs.

    Parameters
    ----------
    sim : Sim
      The Sim instance which will manage this UE.
    xyz : [float, float, float]
      Position of UE in metres, and antenna height.
    h_UT : float
      Antenna height of user terminal in metres; only used if xyz is not provided.
    reporting_interval : float
      Time interval between UE reports being sent to the serving cell.
    f_callback :
      A function with signature ``f_callback(self,kwargs)``, which will be called at each iteration of the main loop.
    f_callback_kwargs :
      kwargs for previous function.
    pathloss_model
      An instance of a pathloss model.  This must be a callable object which
      takes two arguments, each a 3-vector.  The first represent the transmitter
      location, and the second the receiver location.  It must return the
      pathloss in dB along this signal path.
      If set to ``None`` (the default), a standard urban macrocell model
      is used.
      See further ``NR_5G_standard_functions_00.py``.
  '''
  i=0

  def __init__(s,sim,xyz=None,reporting_interval=1.0,pathloss_model=None,h_UT=2.0,f_callback=None,f_callback_kwargs={},verbosity=0):
    s.sim=sim
    s.i=UE.i; UE.i+=1
    s.serving_cell=None
    s.f_callback=f_callback
    s.f_callback_kwargs=f_callback_kwargs
    # next will be a record of last 10 serving cell ids,
    # with time of last attachment.
    # 0=>current, 1=>previous, etc. -1 => not valid)
    # This is for use in handover algorithms
    s.serving_cell_ids=deque([(-1,None)]*10,maxlen=10)
    s.reporting_interval=reporting_interval
    if xyz is not None:
      s.xyz=np.array(xyz,dtype=float)
    else:
      s.xyz=250.0+500.0*s.sim.rng.random(3)
      s.xyz[2]=h_UT
    if verbosity>1: print(f'UE[{s.i}]   is at',s.xyz,file=stderr)
    # We assume here that the UMa_pathloss model needs to be instantiated,
    # but other user-provided models are already instantiated,
    # and provide callable objects...
    if pathloss_model is None:
      s.pathloss=UMa_pathloss(fc_GHz=s.sim.params['fc_GHz'],h_UT=s.sim.params['h_UT'],h_BS=s.sim.params['h_BS'])
      if verbosity>1: print(f'Using 5G standard urban macrocell pathloss model.',file=stderr)
    else:
      s.pathloss=pathloss_model
      if s.pathloss.__doc__ is not None:
        if verbosity>1: print(f'Using user-specified pathloss model "{s.pathloss.__doc__}".',file=stderr)
      else:
        print(f'Using user-specified pathloss model.',file=stderr)
    s.verbosity=verbosity
    s.noise_power_dBm=-140.0
    s.cqi=None
    s.sinr_dB=None
    # Keith Briggs 2022-10-12 loops now started in Sim.__init__
    #s.sim.env.process(s.run_subband_cqi_report())
    #s.sim.env.process(s.loop()) # this does reports to all cells

  def __repr__(s):
    return f'UE(index={s.i},xyz={s.xyz},serving_cell={s.serving_cell})'

  def set_f_callback(s,f_callback,**kwargs):
    ' Add a callback function to the main loop of this UE '
    s.f_callback=f_callback
    s.f_callback_kwargs=kwargs

  def loop(s):
    ' Main loop of UE class '
    if s.verbosity>1:
      print(f'Main loop of UE[{s.i}] started')
      stdout.flush()
    while True:
      if s.f_callback is not None: s.f_callback(s,**s.f_callback_kwargs)
      s.send_rsrp_reports()
      s.send_subband_cqi_report() # FIXME merge these two reports
      #print(f'dbg: Main loop of UE class started'); exit()
      yield s.sim.env.timeout(s.reporting_interval)

  def get_serving_cell(s):
    '''
    Return the current serving Cell object (not index) for this UE instance.
    '''
    ss=s.serving_cell
    if ss is None: return None
    return s.serving_cell

  def get_serving_cell_i(s):
    '''
    Return the current serving Cell index for this UE instance.
    '''
    ss=s.serving_cell
    if ss is None: return None
    return s.serving_cell.i

  def get_xyz(s):
    '''
    Return the current position of this UE.
    '''
    return s.xyz

  def set_xyz(s,xyz,verbose=False):
    '''
    Set a new position for this UE.
    '''
    s.xyz=np.array(xyz)
    if verbose: print(f'UE[{s.i}] is now at {s.xyz}',file=stderr)

  def attach(s,cell,quiet=True):
    '''
    Attach this UE to a specific Cell instance.
    '''
    cell.attached.add(s.i)
    s.serving_cell=cell
    s.serving_cell_ids.appendleft((cell.i,s.sim.env.now,))
    if not quiet and s.verbosity>0:
      print(f'UE[{s.i:2}] is attached to cell[{cell.i}]',file=stderr)

  def detach(s,quiet=True):
    '''
    Detach this UE from its serving cell.
    '''
    if s.serving_cell is None:  # Keith Briggs 2022-08-08 added None test
      return
    s.serving_cell.attached.remove(s.i)
    # clear saved reports from this UE...
    reports=s.serving_cell.reports
    for x in reports:
      if s.i in reports[x]: del reports[x][s.i]
    if not quiet and s.verbosity>0:
      print(f'UE[{s.i}] detached from cell[{s.serving_cell.i}]',file=stderr)
    s.serving_cell=None

  def attach_to_strongest_cell_simple_pathloss_model(s):
    '''
    Attach to the cell delivering the strongest signal
    at the current UE position. Intended for initial attachment only.
    Uses only a simple power-law pathloss model.  For proper handover
    behaviour, use the MME module.
    '''
    celli=s.sim.get_strongest_cell_simple_pathloss_model(s.xyz)
    s.serving_cell=s.sim.cells[celli]
    s.serving_cell.attached.add(s.i)
    if s.verbosity>0:
      print(f'UE[{s.i:2}] ⟵⟶  cell[{celli}]',file=stderr)

  def attach_to_nearest_cell(s):
    '''
    Attach this UE to the geographically nearest Cell instance.
    Intended for initial attachment only.
    '''
    dmin,celli=_nearest_weighted_point(s.xyz[:2],s.sim.cell_locations[:,:2])
    if 0: # dbg
      print(f'_nearest_weighted_point: celli={celli} dmin={dmin:.2f}')
      for cell in s.sim.cells:
        d=np.linalg.norm(cell.xyz-s.xyz)
        print(f'Cell[{cell.i}] is at distance {d:.2f}')
    s.serving_cell=s.sim.cells[celli]
    s.serving_cell.attached.add(s.i)
    if s.verbosity>0:
      print(f'UE[{s.i:2}] ⟵⟶  cell[{celli}]',file=stderr)

  def get_CQI(s):
    '''
    Return the current CQI of this UE, as an array across all subbands.
    '''
    return s.cqi

  def get_SINR_dB(s):
    '''
    Return the current SINR of this UE, as an array across all subbands.
    The return value ``None`` indicates that there is no current report.
    '''
    return s.sinr_dB

  def send_rsrp_reports(s,threshold=-120.0):
    '''
    Send RSRP reports in dBm to all cells for which it is over the threshold.
    Subbands not handled.
    '''
    # antenna pattern computation added Keith Briggs 2021-11-24.
    for cell in s.sim.cells:
      pl_dB=s.pathloss(cell.xyz,s.xyz) # 2021-10-29
      antenna_gain_dB=0.0
      if cell.pattern is not None:
        vector=s.xyz-cell.xyz # vector pointing from cell to UE
        angle_degrees=(180.0/math_pi)*atan2(vector[1],vector[0])
        antenna_gain_dB=cell.pattern(angle_degrees) if callable(cell.pattern) \
          else cell.pattern[int(angle_degrees)%360]
      rsrp_dBm=cell.power_dBm+antenna_gain_dB+cell.MIMO_gain_dB-pl_dB
      rsrp=from_dB(rsrp_dBm)
      if rsrp_dBm>threshold:
        cell.reports['rsrp'][s.i]=(s.sim.env.now,rsrp_dBm)
        if s.i not in cell.rsrp_history:
          cell.rsrp_history[s.i]=deque([-np.inf,]*10,maxlen=10)
        cell.rsrp_history[s.i].appendleft(rsrp_dBm)

  def send_subband_cqi_report(s):
    '''
    For this UE, send an array of CQI reports, one for each subband; and a total throughput report, to the serving cell.
    What is sent is a 2-tuple (current time, array of reports).
    For RSRP reports, use the function ``send_rsrp_reports``.
    Also saves the CQI[1]s in s.cqi, and returns the throughput value.
    '''
    if s.serving_cell is None: return 0.0 # 2022-08-08 detached
    interference=from_dB(s.noise_power_dBm)*np.ones(s.serving_cell.n_RBs)
    for cell in s.sim.cells:
      pl_dB=s.pathloss(cell.xyz,s.xyz)
      antenna_gain_dB=0.0
      if cell.pattern is not None:
        vector=s.xyz-cell.xyz # vector pointing from cell to UE
        angle_degrees=(180.0/math_pi)*atan2(vector[1],vector[0])
        antenna_gain_dB=cell.pattern(angle_degrees) if callable(cell.pattern) \
          else cell.pattern[int(angle_degrees)%360]
      if cell.i==s.serving_cell.i: # wanted signal
        rsrp_dBm=cell.MIMO_gain_dB+antenna_gain_dB+cell.power_dBm-pl_dB
      else: # unwanted interference
        received_interference_power=antenna_gain_dB+cell.power_dBm-pl_dB
        interference+=from_dB(received_interference_power)*cell.subband_mask
    rsrp=from_dB(rsrp_dBm)
    s.sinr_dB=to_dB(rsrp/interference) # scalar/array
    s.cqi=cqi=SINR_to_CQI(s.sinr_dB)
    spectral_efficiency=np.array([CQI_to_64QAM_efficiency(cqi_i) for cqi_i in cqi])
    now=float(s.sim.env.now)
    # per-UE throughput...
    throughput_Mbps=s.serving_cell.bw_MHz*(spectral_efficiency@s.serving_cell.subband_mask)/s.serving_cell.n_RBs/len(s.serving_cell.attached)
    s.serving_cell.reports['cqi'][s.i]=(now,cqi)
    s.serving_cell.reports['throughput_Mbps'][s.i]=(now,throughput_Mbps,)
    return throughput_Mbps

  def run_subband_cqi_report(s): # FIXME merge this with rsrp reporting
    while True:
      #if s.serving_cell is not None: # UE must be attached 2022-08-08
        s.send_subband_cqi_report()
        yield s.sim.env.timeout(s.reporting_interval)

# END class UE

class Sim:
  '''
  Class representing the complete simulation.

  Parameters
  ----------
  params : dict
    A dictionary of additional global parameters which need to be accessible to downstream functions. In the instance, these parameters will be available as ``sim.params``.  If ``params['profile']`` is set to a non-empty string, then a code profile will be performed and the results saved to the filename given by the string.  There will be some execution time overhead when profiling.
  '''

  def __init__(s,params={'fc_GHz':3.5,'h_UT':2.0,'h_BS':20.0},show_params=True,rng_seed=0):
    s.__version__=__version__
    s.params=params
    # set default values for operating frequenct, user terminal height, and
    # base station height...
    if 'fc_GHz' not in params: params['fc_GHz']=3.5
    if 'h_UT'   not in params: params['h_UT']=2.0
    if 'h_BS'   not in params: params['h_BS']=20.0
    s.env=simpy.Environment()
    s.rng=np.random.default_rng(rng_seed)
    s.loggers=[]
    s.scenario=None
    s.ric=None
    s.mme=None
    s.hetnet=None # unknown at this point; will be set to True or False
    s.cells=[]
    s.UEs=[]
    s.events=[]
    s.cell_locations=np.empty((0,3))
    np.set_printoptions(precision=2,linewidth=200)
    pyv=pyversion.replace('\n','') #[:pyversion.index('(default')]
    print(f'python version={pyv}',file=stderr)
    print(f'numpy  version={np.__version__}',file=stderr)
    print(f'simpy  version={simpy.__version__}',file=stderr)
    print(f'AIMM simulator version={s.__version__}',file=stderr)
    if show_params:
      print(f'Simulation parameters:',file=stderr)
      for param in s.params:
        print(f"  {param}={s.params[param]}",file=stderr)

  def _set_hetnet(s):
    #  internal function only - decide whether we have a hetnet
    powers=set(cell.get_power_dBm() for cell in s.cells)
    s.hetnet=len(powers)>1 # powers are not all equal

  def wait(s,interval=1.0):
    '''
    Convenience function to avoid low-level reference to env.timeout().
    ``loop`` functions in each class must yield this.
    '''
    return s.env.timeout(interval)

  def make_cell(s,**kwargs):
    '''
    Convenience function: make a new Cell instance and add it to the simulation; parameters as for the Cell class. Return the new Cell instance.  It is assumed that Cells never move after being created (i.e. the initial xyz[1] stays the same throughout the simulation).
    '''
    s.cells.append(Cell(s,**kwargs))
    xyz=s.cells[-1].get_xyz()
    s.cell_locations=np.vstack([s.cell_locations,xyz])
    return s.cells[-1]

  def make_UE(s,**kwargs):
    '''
    Convenience function: make a new UE instance and add it to the simulation; parameters as for the UE class. Return the new UE instance.
    '''
    s.UEs.append(UE(s,**kwargs))
    return s.UEs[-1]

  def get_ncells(s):
    '''
    Return the current number of cells in the simulation.
    '''
    return len(s.cells)

  def get_nues(s):
    '''
    Return the current number of UEs in the simulation.
    '''
    return len(s.UEs)

  def get_UE_position(s,ue_i):
    '''
    Return the xyz position of UE[i] in the simulation.
    '''
    return s.UEs[ue_i].xyz

  def get_average_throughput(s):
    '''
    Return the average throughput over all UEs attached to all cells.
    '''
    ave,k=0.0,0
    for cell in s.cells:
      k+=1
      ave+=(cell.get_average_throughput()-ave)/k
    return ave

  def add_logger(s,logger):
    '''
    Add a logger to the simulation.
    '''
    assert isinstance(logger,Logger)
    s.loggers.append(logger)

  def add_loggers(s,loggers):
    '''
    Add a sequence of loggers to the simulation.
    '''
    for logger in loggers:
      assert isinstance(logger,Logger)
      s.loggers.append(logger)

  def add_scenario(s,scenario):
    '''
    Add a Scenario instance to the simulation.
    '''
    assert isinstance(scenario,Scenario)
    s.scenario=scenario

  def add_ric(s,ric):
    '''
    Add a RIC instance to the simulation.
    '''
    assert isinstance(ric,RIC)
    s.ric=ric

  def add_MME(s,mme):
    '''
    Add an MME instance to the simulation.
    '''
    assert isinstance(mme,MME)
    s.mme=mme

  def add_event(s,event):
    s.events.append(event)

  def get_serving_cell(s,ue_i):
    if ue_i<len(s.UEs): return s.UEs[ue_i].serving_cell
    return None

  def get_serving_cell_i(s,ue_i):
    if ue_i<len(s.UEs): return s.UEs[ue_i].serving_cell.i
    return None

  def get_nearest_cell(s,xy):
    '''
    Return the index of the geographical nearest cell (in 2 dimensions)
    to the point xy.
    '''
    return _nearest_weighted_point(xy[:2],s.cell_locations[:,:2],w=1.0)[1]

  def get_strongest_cell_simple_pathloss_model(s,xyz,alpha=3.5):
    '''
    Return the index of the cell delivering the strongest signal
    at the point xyz (in 3 dimensions), with pathloss exponent alpha.
    Note: antenna pattern is not used, so this function is deprecated,
    but is adequate for initial UE attachment.
    '''
    p=np.array([from_dB(cell.get_power_dBm()) for cell in s.cells])
    return _nearest_weighted_point(xyz,s.cell_locations,w=p**(-1.0/alpha))[1]

  def get_best_rsrp_cell(s,ue_i,dbg=False):
    '''
    Return the index of the cell delivering the highest RSRP at UE[i].
    Relies on UE reports, and ``None`` is returned if there are not enough
    reports (yet) to determine the desired output.
    '''
    k,best_rsrp=None,-np.inf
    cell_rsrp_reports=dict((cell.i,cell.reports['rsrp']) for cell in s.cells)
    for cell in s.cells:
      if ue_i not in cell_rsrp_reports[cell.i]: continue # no reports for this UE
      time,rsrp=cell_rsrp_reports[cell.i][ue_i] # (time, subband reports)
      if dbg: print(f"get_best_rsrp_cell at {float(s.env.now):.0f}: cell={cell.i} UE={ue_i} rsrp=",rsrp,file=stderr)
      ave_rsrp=np.average(rsrp) # average RSRP over subbands
      if ave_rsrp>best_rsrp: k,best_rsrp=cell.i,ave_rsrp
    return k

  def _start_loops(s):
    # internal use only - start all main loops
    for logger in s.loggers:
      s.env.process(logger.loop())
    if s.scenario is not None:
      s.env.process(s.scenario.loop())
    if s.ric is not None:
      s.env.process(s.ric.loop())
    if s.mme is not None:
      s.env.process(s.mme.loop())
    for event in s.events: # TODO ?
      s.env.process(event)
    for cell in s.cells: # 2022-10-12 start Cells
      s.env.process(cell.loop())
    for ue in s.UEs: # 2022-10-12 start UEs
      #print(f'About to start main loop of UE[{ue.i}]..')
      s.env.process(ue.loop())
      #s.env.process(UE.run_subband_cqi_report())
    #sleep(2); exit()

  def run(s,until):
    s._set_hetnet()
    s.until=until
    print(f'Sim: starting run for simulation time {until} seconds...',file=stderr)
    s._start_loops()
    t0=time()
    if 'profile' in s.params and s.params['profile']:
      # https://docs.python.org/3.6/library/profile.html
      # to keep python 3.6 compatibility, we don't use all the
      # features for profiling added in 3.8 or 3.9.
      profile_filename=s.params['profile']
      print(f'profiling enabled: output file will be {profile_filename}.',file=stderr)
      import cProfile,pstats,io
      pr=cProfile.Profile()
      pr.enable()
      s.env.run(until=until) # this is what is profiled
      pr.disable()
      strm=io.StringIO()
      ps=pstats.Stats(pr,stream=strm).sort_stats('tottime')
      ps.print_stats()
      tbl=strm.getvalue().split('\n')
      profile_file=open(profile_filename,'w')
      for line in tbl[:50]: print(line,file=profile_file)
      profile_file.close()
      print(f'profile written to {profile_filename}.',file=stderr)
    else:
      s.env.run(until=until)
    print(f'Sim: finished main loop in {(time()-t0):.2f} seconds.',file=stderr)
    #print(f'Sim: hetnet={s.hetnet}.',file=stderr)
    if s.mme is not None:
      s.mme.finalize()
    if s.ric is not None:
      s.ric.finalize()
    for logger in s.loggers:
      logger.finalize()

# END class Sim

class Scenario:

  '''
    Base class for a simulation scenario. The default does nothing.

    Parameters
    ----------
    sim : Sim
      Simulator instance which will manage this Scenario.
    func : function
      Function called to perform actions.
    interval : float
      Time interval between actions.
    verbosity : int
      Level of debugging output (0=none).

  '''

  def __init__(s,sim,func=None,interval=1.0,verbosity=0):
    s.sim=sim
    s.func=func
    s.verbosity=verbosity
    s.interval=interval

  def loop(s):
    '''
    Main loop of Scenario class.  Should be overridden to provide different functionalities.
    '''
    while True:
      if s.func is not None: s.func(s.sim)
      yield s.sim.env.timeout(s.interval)

# END class Scenario

class Logger:

  '''
  Represents a simulation logger. Multiple loggers (each with their own file) can be used if desired.

  Parameters
  ----------
    sim : Sim
      The Sim instance which will manage this Logger.
    func : function
      Function called to perform logginf action.
    header : str
      Arbitrary text to write to the top of the logfile.
    f : file object
      An open file object which will be written or appended to.
    logging_interval : float
      Time interval between logging actions.

  '''
  def __init__(s,sim,func=None,header='',f=stdout,logging_interval=10,np_array_to_str=np_array_to_str):
    s.sim=sim
    s.func=s.default_logger if func is None else func
    s.f=f
    s.np_array_to_str=np_array_to_str
    s.logging_interval=float(logging_interval)
    if header: s.f.write(header)

  def default_logger(s,f=stdout):
    for cell in s.sim.cells:
      for ue_i in cell.reports['cqi']:
        rep=cell.reports['cqi'][ue_i]
        if rep is None: continue
        cqi=s.np_array_to_str(rep[1])
        f.write(f'{cell.i}\t{ue_i}\t{cqi}\n')

  def loop(s):
    '''
    Main loop of Logger class.
    Can be overridden to provide custom functionality.
    '''
    while True:
      s.func(f=s.f)
      yield s.sim.env.timeout(s.logging_interval)

  def finalize(s):
    '''
    Function called at end of simulation, to implement any required finalization actions.
    '''
    pass

# END class Logger

class MME:
  '''
  Represents a MME, for handling UE handovers.

  Parameters
  ----------
    sim : Sim
      Sim instance which will manage this Scenario.
    interval : float
      Time interval between checks for handover actions.
    verbosity : int
      Level of debugging output (0=none).
    strategy : str
      Handover strategy; possible values are ``strongest_cell_simple_pathloss_model`` (default), or ``best_rsrp_cell``.
    anti_pingpong : float
      If greater than zero, then a handover pattern x->y->x between cells x and y is not allowed within this number of seconds. Default is 0.0, meaning pingponging is not suppressed.
  '''

  def __init__(s,sim,interval=10.0,strategy='strongest_cell_simple_pathloss_model',anti_pingpong=30.0,verbosity=0):
    s.sim=sim
    s.interval=interval
    s.strategy=strategy
    s.anti_pingpong=anti_pingpong
    s.verbosity=verbosity
    print(f'MME: using handover strategy {s.strategy}.',file=stderr)

  def do_handovers(s):
    '''
    Check whether handovers are required, and do them if so.
    Normally called from loop(), but can be called manually if required.
    '''
    for ue in s.sim.UEs:
      if ue.serving_cell is None: continue # no handover needed for this UE. 2022-08-08 added None test
      oldcelli=ue.serving_cell.i # 2022-08-26
      CQI_before=ue.serving_cell.get_UE_CQI(ue.i)
      previous,tm=ue.serving_cell_ids[1]
      if s.strategy=='strongest_cell_simple_pathloss_model':
        celli=s.sim.get_strongest_cell_simple_pathloss_model(ue.xyz)
      elif s.strategy=='best_rsrp_cell':
        celli=s.sim.get_best_rsrp_cell(ue.i)
        if celli is None:
          celli=s.sim.get_strongest_cell_simple_pathloss_model(ue.xyz)
      else:
        print(f'MME.loop: strategy {s.strategy} not implemented, quitting!',file=stderr)
        exit()
      if celli==ue.serving_cell.i: continue
      if s.anti_pingpong>0.0 and previous==celli:
        if s.sim.env.now-tm<s.anti_pingpong:
          if s.verbosity>2:
            print(f't={float(s.sim.env.now):8.2f} handover of UE[{ue.i}] suppressed by anti_pingpong heuristic.',file=stderr)
          continue # not enough time since we were last on this cell
      ue.detach(quiet=True)
      ue.attach(s.sim.cells[celli])
      ue.send_rsrp_reports() # make sure we have reports immediately
      ue.send_subband_cqi_report()
      if s.verbosity>1:
        CQI_after=ue.serving_cell.get_UE_CQI(ue.i)
        print(f't={float(s.sim.env.now):8.2f} handover of UE[{ue.i:3}] from Cell[{oldcelli:3}] to Cell[{ue.serving_cell.i:3}]',file=stderr,end=' ')
        print(f'CQI change {CQI_before} -> {CQI_after}',file=stderr)

  def loop(s):
    '''
    Main loop of MME.
    '''
    yield s.sim.env.timeout(0.5*s.interval) # stagger the intervals
    print(f'MME started at {float(s.sim.env.now):.2f}, using strategy="{s.strategy}" and anti_pingpong={s.anti_pingpong:.0f}.',file=stderr)
    while True:
      s.do_handovers()
      yield s.sim.env.timeout(s.interval)

  def finalize(s):
    '''
    Function called at end of simulation, to implement any required finalization actions.
    '''
    pass

# END class MME

class RIC:
  '''
  Base class for a RIC, for hosting xApps.  The default does nothing.

  Parameters
  ----------
    sim : Sim
      Simulator instance which will manage this Scenario.
    interval : float
      Time interval between RIC actions.
    verbosity : int
      Level of debugging output (0=none).
  '''

  def __init__(s,sim,interval=10,verbosity=0):
    s.sim=sim
    s.interval=interval
    s.verbosity=verbosity

  def finalize(s):
    '''
    Function called at end of simulation, to implement any required finalization actions.
    '''
    pass

  def loop(s):
    '''
    Main loop of RIC class.  Must be overridden to provide functionality.
    '''
    print(f'RIC started at {float(s.sim.env.now):.2}.',file=stderr)
    while True:
      yield s.sim.env.timeout(s.interval)
# END class RIC

if __name__=='__main__': # a simple self-test

  np.set_printoptions(precision=4,linewidth=200)

  class MyLogger(Logger):
    def loop(s):
      while True:
        for cell in s.sim.cells:
          if cell.i!=0: continue # cell[0] only
          for ue_i in cell.reports['cqi']:
            if ue_i!=0: continue # UE[0] only
            rep=cell.reports['cqi'][ue_i]
            if not rep: continue
            xy= s.np_array_to_str(s.sim.UEs[ue_i].xyz[:2])
            cqi=s.np_array_to_str(cell.reports['cqi'][ue_i][1])
            tp= s.np_array_to_str(cell.reports['throughput_Mbps'][ue_i][1])
            s.f.write(f'{s.sim.env.now:.1f}\t{xy}\t{cqi}\t{tp}\n')
        yield s.sim.env.timeout(s.logging_interval)

  def test_01(ncells=4,nues=9,n_RBs=2,until=1000.0):
    sim=Sim()
    for i in range(ncells):
      sim.make_cell(n_RBs=n_RBs,MIMO_gain_dB=3.0,verbosity=0)
    sim.cells[0].set_xyz((500.0,500.0,20.0)) # fix cell[0]
    for i in range(nues):
      ue=sim.make_UE(verbosity=1)
      if 0==i: # force ue[0] to attach to cell[0]
        ue.set_xyz([501.0,502.0,2.0],verbose=True)
      ue.attach_to_nearest_cell()
    scenario=Scenario(sim,verbosity=0)
    logger=MyLogger(sim,logging_interval=1.0)
    ric=RIC(sim)
    sim.add_logger(logger)
    sim.add_scenario(scenario)
    sim.add_ric(ric)
    sim.run(until=until)

  test_01()
