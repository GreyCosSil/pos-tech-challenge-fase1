# import shutil
# import pandas as pd
# pd.set_option('display.float_format', lambda x: '%.2f' % x)
# pd.set_option('display.max_columns', None)
# import numpy as np
# import os
# import kagglehub
# from pathlib import Path


import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
import numpy as np
import os
from pathlib import Path
import shutil
import kagglehub


path = kagglehub.dataset_download("blastchar/telco-customer-churn")

df = pd.read_csv(Path(path) / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"Shape of the dataset: {df.shape[0]} rows and {df.shape[1]} columns")