years = ["2019", "2020", "2021", "2022", "2023"]
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
year_month = []
for year in years:
    year_month += [[(year, month) for month in months]]
    
print(year_month)
