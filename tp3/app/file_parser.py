import csv
from settings import settings

def parse_csv():
  with open(f"{settings.Config.data_path}/regression_data.csv", newline='') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    result = []
    for row in csv_reader:
        result.append(row)
    return result