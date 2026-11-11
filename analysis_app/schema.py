FIELDS = {'observation_id': 'str', 'device': 'str', 'batch': 'str', 'cycle': 'int', 'vibration': 'float', 'temperature': 'float', 'load': 'float', 'failed': 'int', 'recorded': 'int'}
REQUIRED = ['observation_id', 'device', 'batch', 'cycle', 'vibration', 'temperature', 'load', 'failed', 'recorded']
ENTITY = 'device'
GROUP = 'batch'
MEASURE = 'vibration'
PRODUCT = 'cohort-feature-selection'
UNIT = '毫米每秒'
