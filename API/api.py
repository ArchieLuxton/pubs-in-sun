from flask import Flask, request
import requests

app = Flask(__name__)

@app.route("/building", methods=["GET"])
def building():
    address = request.args.get("address")
    # Convert the address to latitude and longitude using a geocoding service

    url = "http://overpass-api.de/api/interpreter"

    query = """[out:json];
    node(around:50,LATITUDE,LONGITUDE)["building"];
    out body;"""

    response = requests.get(url, params={"data": query})

    data = response.json()

    return data

if __name__ == "__main__":
    app.run(debug=True)
