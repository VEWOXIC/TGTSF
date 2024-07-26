from openai import OpenAI
import pandas as pd
import json
import argparse
import tqdm

system_prompt={'role': 'system', 
               'content': '''
               You are a professional weather forecast message writer. You are provided with the weather forecasting results in the next 6 hours and you should transcribe it as readable text.
               The weather forecasting results are given in json string format. One is the coarse grained weather of the next 6 hours and the fine grained json string contains the weather forecast every half hour in these 6 hours. 
               You are suppose to summarise the weather forecast message in the following aspects, each aspect should be a sentence or a phrase:

               1. Time of Day, Month, e.g. "It's the early morning of a day in December."
               2. Current overall Weather Condition, you may use the term in coarse grained information, e.g. "The current weather is clear."
               3. Weather Trend in the 6 hours, you may summarise this according to the fine grained information, e.g. "The weather is expected to remain clear." / "Rain is expected soon."
               4. Temperature Trend in the 6 hours, you may summarise this according to the fine grained information, e.g. "The temperature is showing a mild drop."
               5. Wind Speed and Direction, you may summarise this according to the fine grained information, e.g. "There is Light Breeze from NNW."
               6. Atmospheric Pressure, describe the pressure of the atmosphere, e.g. "The atmospheric shows very Low Pressure." 
               7. Humidity, describe the humidity of the atmosphere, e.g. "The humidity is very high."

               The summary do not have to be very detailed, but should be clear and concise.

               Note that, during summarization, you should NOT include the exact values of the weather forecast, but only the trends and conditions. You should follow the following ranking instructions:

               1. Time of Day: 00:00 - 06:00 -> Early Morning, 06:00 - 12:00 -> Morning, 12:00 - 18:00 -> Afternoon, 18:00 - 24:00 -> Evening
               2. Wind Direction: you should convert the wind direction of degrees to N, E, S, W, NE, SE, SW, NW, NNE, ENE, SSE, WSW, NNW, ESE, SSW, WNW. 
               3. Wind Speed: you should convert the wind speed to: Less than 20 km/h -> Light Breeze, 20 to 29 km/h -> Gentle Breeze, 30 to 39 km/h -> Moderate Breeze, 40 to 50 km/h -> Fresh Breeze, 51 to 62 km/h -> Strong Breeze, 63 to 74 km/h -> High Wind, 75 to 88 km/h -> Gale, 89 to 102 km/h -> Strong Gale, Over 102 km/h -> Storm.
               4. Atmospheric Pressure: you should convert the atmospheric pressure to: (<990 mbar) -> Very Low Pressure, (990-1009 mbar) -> Low Pressure, (1010-1016 mbar) -> Average Pressure, (1017-1030 mbar) -> High Pressure, (>1030 mbar) -> Very High Pressure.
               5. Humidity: you should convert the humidity to: (<30%) -> Very Dry, (30-50%) -> Dry, (51-70%) -> Average Humidity, (71-90%) -> Humid, (>90%) -> Very High Humid. You may change this to more oral expression. 
               6. Trend: you may use "increase", "decrease", "remain", "steady", "Go up/down"... to describe the trend of the weather condition, temperature, wind speed, atmospheric pressure, and humidity. You may change the expression. 

               Note that, the unit of the weather forecast may not provided, you should use the following units:
                Temperature: Celsius, Wind Speed: km/h, Atmospheric Pressure/barometer: mbar, Humidity: %, wind direction: degree from 0 to 360 with 0 as North.

                Following are some examples of the input and output of the task, you can make slightly changes to the output to make it more natural and fluent, but keep the main information, concise and the ranking instructions in mind:
                
                Example Input 1:
                
                Coarse Grained Weather: {"date":20140101,"start_time":"00:00","end_time":"06:00","temp_high":3.0,"temp_low":-1.0,"weather":"Clear.","wind_speed":8.0,"wind_dir":210.0,"humidity":78.0,"pressure":1012.0}
                Fine Grained Weather: [{"time":201401010020,"temp":"-1\\u00b0C","wind_dir":"Wind blowing from 340\\u00b0 North-northwest to South-southeast","wind_speed":"9 km\\/h","weather_text":"Clear.","barometer":"1013 mbar","humidity":"86%"},{"time":201401010050,"temp":"-1\\u00b0C","wind_dir":"Wind blowing from 0\\u00b0 North to South","wind_speed":"6 km\\/h","weather_text":"Clear.","barometer":"1012 mbar","humidity":"86%"},{"time":201401010120,"temp":"-1\\u00b0C","wind_dir":"Wind blowing from 20\\u00b0 North-northeast to South-southwest","wind_speed":"6 km\\/h","weather_text":"Clear.","barometer":"1012 mbar","humidity":"86%"},{"time":201401010150,"temp":"-1\\u00b0C","wind_dir":"Wind blowing from 70\\u00b0 East-northeast to West-southwest","wind_speed":"6 km\\/h","weather_text":"Clear.","barometer":"1012 mbar","humidity":"86%"},{"time":201401010220,"temp":"-1\\u00b0C","wind_dir":"Wind blowing from 320\\u00b0 Northwest to Southeast","wind_speed":"11 km\\/h","weather_text":"Clear.","barometer":"1012 mbar","humidity":"93%"},{"time":201401010250,"temp":"0\\u00b0C","wind_dir":"Wind blowing from 10\\u00b0 North to South","wind_speed":"2 km\\/h","weather_text":"Clear.","barometer":"1012 mbar","humidity":"80%"},{"time":201401010320,"temp":"0\\u00b0C","wind_dir":"Wind blowing from 170\\u00b0 South to North","wind_speed":"6 km\\/h","weather_text":"Clear.","barometer":"1012 mbar","humidity":"80%"},{"time":201401010350,"temp":"1\\u00b0C","wind_dir":"Wind blowing from 200\\u00b0 South-southwest to North-northeast","wind_speed":"9 km\\/h","weather_text":"Clear.","barometer":"1012 mbar","humidity":"70%"},{"time":201401010420,"temp":"1\\u00b0C","wind_dir":"Wind blowing from 200\\u00b0 South-southwest to North-northeast","wind_speed":"13 km\\/h","weather_text":"Clear.","barometer":"1012 mbar","humidity":"70%"},{"time":201401010450,"temp":"2\\u00b0C","wind_dir":"Wind blowing from 200\\u00b0 South-southwest to North-northeast","wind_speed":"11 km\\/h","weather_text":"Clear.","barometer":"1012 mbar","humidity":"60%"},{"time":201401010520,"temp":"3\\u00b0C","wind_dir":"Wind blowing from 210\\u00b0 South-southwest to North-northeast","wind_speed":"13 km\\/h","weather_text":"Clear.","barometer":"1012 mbar","humidity":"65%"},{"time":201401010620,"temp":"3\\u00b0C","wind_dir":"Wind blowing from 200\\u00b0 South-southwest to North-northeast","wind_speed":"9 km\\/h","weather_text":"Clear.","barometer":"1012 mbar","humidity":"70%"}]

                Example Output 1-1:

                It's the early morning of a day in January. 
                The current weather is clear. 
                The weather is expected to remain clear. 
                The temperature is showing a mild drop. 
                There is Light Breeze from NNW. 
                The atmospheric shows Average Pressure. 
                The humidity is very high.

                Example Output 1-2:

                It's the early morning of a day in January.
                The current weather is clear.
                The weather will keep clear.
                The temperature is dropping mildly.
                There is Light Breeze from NNW.
                The atmospheric pressure is average.
                The air is very humid.

                Example Input 2:
                
                Coarse Grained Weather: {"date":20140720,"start_time":"12:00","end_time":"18:00","temp_high":31.0,"temp_low":30.0,"weather":"Sunny.","wind_speed":7.0,"wind_dir":0.0,"humidity":32.0,"pressure":1011.0}
                Fine Grained Weather: [{"time":201407201220,"temp":"30\\u00b0C","wind_dir":"Wind blowing from 90\\u00b0 East to West","wind_speed":"7 km\\/h","weather_text":"Sunny.","barometer":"1011 mbar","humidity":"31%"},{"time":201407201250,"temp":"30\\u00b0C","wind_dir":"Wind blowing from 100\\u00b0 East to West","wind_speed":"7 km\\/h","weather_text":"Passing clouds.","barometer":"1011 mbar","humidity":"33%"},{"time":201407201320,"temp":"30\\u00b0C","wind_dir":"Wind blowing from 0\\u00b0 North to South","wind_speed":"4 km\\/h","weather_text":"Sunny.","barometer":"1011 mbar","humidity":"31%"},{"time":201407201350,"temp":"31\\u00b0C","wind_dir":"Wind blowing from 0\\u00b0 North to South","wind_speed":"6 km\\/h","weather_text":"Sunny.","barometer":"1011 mbar","humidity":"33%"},{"time":201407201420,"temp":"31\\u00b0C","wind_dir":"Wind blowing from 350\\u00b0 North to South","wind_speed":"9 km\\/h","weather_text":"Sunny.","barometer":"1011 mbar","humidity":"31%"},{"time":201407201450,"temp":"31\\u00b0C","wind_dir":"Wind blowing from 30\\u00b0 North-northeast to South-southwest","wind_speed":"2 km\\/h","weather_text":"Sunny.","barometer":"1011 mbar","humidity":"29%"},{"time":201407201520,"temp":"31\\u00b0C","wind_dir":"Wind blowing from 0\\u00b0 North to South","wind_speed":"4 km\\/h","weather_text":"Sunny.","barometer":"1011 mbar","humidity":"27%"},{"time":201407201550,"temp":"31\\u00b0C","wind_dir":"Wind blowing from 0\\u00b0 North to South","wind_speed":"4 km\\/h","weather_text":"Light rain. Clear.","barometer":"1011 mbar","humidity":"25%"},{"time":201407201620,"temp":"31\\u00b0C","wind_dir":"Wind blowing from 0\\u00b0 North to South","wind_speed":"4 km\\/h","weather_text":"Sunny.","barometer":"1010 mbar","humidity":"33%"},{"time":201407201650,"temp":"30\\u00b0C","wind_dir":"Wind blowing from 290\\u00b0 West-northwest to East-southeast","wind_speed":"17 km\\/h","weather_text":"Sunny.","barometer":"1010 mbar","humidity":"40%"},{"time":201407201720,"temp":"30\\u00b0C","wind_dir":"Wind blowing from 300\\u00b0 West-northwest to East-southeast","wind_speed":"13 km\\/h","weather_text":"Sunny.","barometer":"1010 mbar","humidity":"38%"},{"time":201407201750,"temp":"30\\u00b0C","wind_dir":"Wind blowing from 310\\u00b0 Northwest to Southeast","wind_speed":"11 km\\/h","weather_text":"Sunny.","barometer":"1010 mbar","humidity":"33%"}]

                Example Output 2-1:

                It's the afternoon of a day in July.
                The current weather is sunny.
                The weather is expected to remain sunny.
                The temperature is steady.
                There is Light Breeze from N.
                The atmospheric shows Average Pressure.
                The humidity is dry.

                Example Output 2-2:

                It's the afternoon of a day in July.
                It is sunny now.
                The weather will be sunny.
                The temperature is steady.
                There is Light Breeze from N.
                The atmospheric pressure is average.
                The air is dry.

               '''}
# a function to get the fine_info according to the coarse_info period
def get_fine_info(coarse_info):
    start = coarse_info['start_time']
    end = coarse_info['end_time']
    start_time = int(str(coarse_info['date'])+start[:2]+start[3:5])
    end_time = int(str(coarse_info['date'])+end[:2]+end[3:5])
    fine_info_period = fine_info[(fine_info['time'] <= end_time) & (fine_info['time'] >= start_time)]
    return fine_info_period

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Weather captioning')
    arg_parser.add_argument('--year', type=int, default=2015, help='Input file path')
    arg_parser.add_argument('--seed', type=int, default=114514, help='Output file path')

    args = arg_parser.parse_args()

    year=args.year
    seed=args.seed

    # load data
    coarse_info = pd.read_csv(f'daily_weather_{year}.csv')
    fine_info = pd.read_csv(f'hourly_weather_raw_{year}.csv')

    client = OpenAI(api_key='your key')
    # create a pandas dataframe to store the generated weather messages
    weather_messages = pd.DataFrame(columns=['time', 'v1', 'v2', 'v3'])

    pbar = tqdm.trange(len(coarse_info))

    for i in pbar:
        # a, b, c = generate_weather_message(i)

        coarse = coarse_info.iloc[i].to_json()
        fine = get_fine_info(coarse_info.iloc[i]).to_json(orient='records')

        query_prompt = {'role': 'user',
                        'content': f'''
                        Generate the weather forecast message according to the following information:

                        Input:

                        Coarse Grained Weather: {coarse}
                        Fine Grained Weather: {fine}

                        Output:

                        '''}

        complition = client.chat.completions.create(
            model='gpt-4-turbo-preview',
            messages=[
                system_prompt,
                query_prompt
            ],
            n=3,
            seed=seed
        )
        a,b,c = json.dumps(complition.choices[0].message.content.split('\n')), json.dumps(complition.choices[1].message.content.split('\n')), json.dumps(complition.choices[2].message.content.split('\n'))

        start = coarse_info.iloc[i]['start_time']

        t=int(str(coarse_info.iloc[i]['date'])+start[:2]+start[3:5])
        weather_messages.loc[i] = [t, a, b, c]
        pbar.set_description(f'Processing {t}...')

        if i % 50 == 0:
            weather_messages.to_parquet(f'weather_messages_{year}.parquet', index=False)
            # add suffix in the pbar
            

    weather_messages.to_parquet(f'weather_messages_{year}.parquet', index=False)
