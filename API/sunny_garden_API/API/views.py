import os

import requests
from django.http import JsonResponse

API_KEY = os.getenv('GOOGLE_API_KEY')


def search_pubs_within_radius(request):
    """
    A view that queries the Google Places API for pubs within a certain radius of a location.

    Can be queried from the front-end search box to populate the list of pubs

    Parameters:
    - request: the HTTP request object
    - latitude: the latitude of the location to search around
    - longitude: the longitude of the location to search around
    - radius: the radius of the search in meters (default is 1000 meters)

    Returns:
    - A JSON response containing a list of pubs, each represented as a dictionary with 'name' and 'address' keys.
    """
    latitude = request.GET.get('latitude')
    longitude = request.GET.get('longitude')
    radius = request.GET.get('radius', 1000)  # default radius is 1000 meters

    # Make a request to the Google Places API
    url = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latitude},{longitude}&radius={radius}&type=bar&key={API_KEY}'
    response = requests.get(url)
    results = response.json().get('results')

    # Extract the information we need from the response
    pubs = []
    for result in results:
        pub = {
            'name': result.get('name'),
            'address': result.get('vicinity')
        }
        pubs.append(pub)

    # Return the pubs as a JSON response
    return JsonResponse({'pubs': pubs})


def get_latitude_and_longitude_from_placename(request):
    """
    A view that queries the Google Places API for the latitude and longitude of a place name.

    Parameters:
    - request: the HTTP request object
    - place: the name of the place to search for

    Returns:
    - A JSON response containing the latitude and longitude of the place.
    """
    place = request.GET.get('place')

    # Make a request to the Google Places API
    url = f'https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={place}&inputtype=textquery&fields=geometry&key={API_KEY}'
    response = requests.get(url)
    result = response.json().get('candidates')[0]
    geometry = result.get('geometry')
    location = geometry.get('location')

    # Extract the latitude and longitude from the response
    latitude = location.get('lat')
    longitude = location.get('lng')

    # Return the latitude and longitude as a JSON response
    return JsonResponse({'latitude': latitude, 'longitude': longitude})