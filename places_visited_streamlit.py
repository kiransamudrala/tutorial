from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np 
import streamlit as st

locator = Nominatim(user_agent="myGeocoder")
# location = locator.geocode('Smyrna,GA, USA')
geocode = RateLimiter(locator.geocode, min_delay_seconds=1)

ads = ['ames ia','desmoines ia','chicago il','minneapolis mn','omaha ne','lincoln ne','new york city ny'
       ,'las vegas nv','san francisco ca','san jose ca','columbus oh','fargo nd','niagara falls ny','tampa fl','savannah ga','raleigh nc'
       ,'dallas tx','san juan pr','seattle wa','pensacola fl']
df = pd.DataFrame(ads,columns=['ADDRESS'])
df['location'] = df['ADDRESS'].apply(geocode)
df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
df[['lat', 'lon', 'altitude']] = pd.DataFrame(df['point'].tolist(), index=df.index)

# Adding code so we can have map default to the center of the data
midpoint = (np.average(df['lat']), np.average(df['lon']))
st.header('Cities I visited:')
st.deck_gl_chart(
            viewport={
                'latitude': midpoint[0],
                'longitude':  midpoint[1],
                'zoom': 2.5
            },
            layers=[{
                'type': 'ScatterplotLayer',
                'data': df,
                'radiusScale': 250,
                'radiusMinPixels': 5,
                'getFillColor': [248, 24, 148],
            }]
        )


ads = ['India','USA','Australia','Spain','France','Italy']
df = pd.DataFrame(ads,columns=['ADDRESS'])
df['location'] = df['ADDRESS'].apply(geocode)
df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
df[['lat', 'lon', 'altitude']] = pd.DataFrame(df['point'].tolist(), index=df.index)

# Adding code so we can have map default to the center of the data
midpoint = (np.average(df['lat']), np.average(df['lon']))
st.header('Countries I have been to:')
st.deck_gl_chart(
            viewport={
                'latitude': midpoint[0],
                'longitude':  midpoint[1],
                'zoom': 0
            },
            layers=[{
                'type': 'ScatterplotLayer',
                'data': df,
                'radiusScale': 250,
                'radiusMinPixels': 5,
                'getFillColor': [248, 24, 148],
            }]
        )

