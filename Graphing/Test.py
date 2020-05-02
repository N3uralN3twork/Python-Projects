from datetime import datetime
import pandas as pd

time = "1749-01"

ts["Date"] = datetime.strptime(, "%Y-%m")

ts["Month"] = pd.to_datetime(ts["Month"], format = "%Y-%m")