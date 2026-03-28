import cdsapi
import time
import os

import multiprocessing


def is_leap_year(y):
    """判断是否为闰年"""
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def download_func(years):
    client = cdsapi.Client(timeout=120)

    for year in [years]:
        if not os.path.exists(".\\02d_ERA5\\" + str(year)):
            os.mkdir(".\\02d_ERA5\\" + str(year))
        for month in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']:
            if month in ('01', '03', '05', '07', '08', '10', '12'):
                days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
                        '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
            elif month in ('04', '06', '09', '11'):
                days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
                        '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
            elif month == '02':
                if is_leap_year(year):
                    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
                            '16',
                            '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29']
                else:
                    days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15',
                            '16',
                            '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']
            else:
                days = []
            for day in days:
                while True:
                    try:
                        # dataset = "derived-era5-single-levels-daily-statistics"
                        # request = {
                        #     "product_type": "reanalysis",
                        #     "variable": ["2m_temperature"],
                        #     'year': str(year),
                        #     'month': [month],
                        #     'day': days,
                        #     "daily_statistic": "daily_maximum",
                        #     "time_zone": "utc+00:00",
                        #     "frequency": "1_hourly"
                        dataset = "reanalysis-era5-single-levels"
                        request = {
                            "product_type": ["reanalysis"],
                            "variable": [
                                "surface_solar_radiation_downwards",
                                "total_sky_direct_solar_radiation_at_surface",
                                "mean_sea_level_pressure"
                            ],
                            "year": [str(year)],
                            "month": [str(month)                            ],
                            "day": [day],
                            "time": [
                                "00:00", "01:00", "02:00",
                                "03:00", "04:00", "05:00",
                                "06:00", "07:00", "08:00",
                                "09:00", "10:00", "11:00",
                                "12:00", "13:00", "14:00",
                                "15:00", "16:00", "17:00",
                                "18:00", "19:00", "20:00",
                                "21:00", "22:00", "23:00"
                            ],
                            "data_format": "netcdf",
                            "download_format": "unarchived",
                            "area": [24.4, 111, 21.4, 115.5]
                        }
                        file_path = '.\\02d_ERA5\\' + str(year) + '\\ERA5-Land' + str(
                            year) + '_' + str(month) + '_' + str(day)+ '.nc'
                        if os.path.exists(file_path): continue
                        client.retrieve(dataset, request, file_path)
                        print(year, month, day)
                    except:
                        print(
                            'Retrying-----------------------------------------------------------------------------------')
                        time.sleep(1)
                        continue
                    else:
                        time.sleep(1)
                        break


if __name__ == "__main__":
    with multiprocessing.Pool(3) as p:
        results = p.map(download_func, range(2015, 2021))
