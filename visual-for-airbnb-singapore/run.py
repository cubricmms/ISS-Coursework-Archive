import os
import csv
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, abort, jsonify
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

engine = create_engine('sqlite:///database.db', echo=False)

current_month =  datetime.now().month - 1 #start from 0
last_month = current_month - 1

ts = pd.read_csv("/home/ubuntu/ts_estimate.csv")
ts_weight = round((ts.iloc[current_month][1] - ts.iloc[last_month][1])/ ts.iloc[last_month][1], 2)
ts_increase = 100 * ts_weight

def load_sqlite():
    for fn in os.listdir('/home/ubuntu/s3data/listing'):
        if ".csv" in fn:
            _fn = os.path.join('/home/ubuntu/s3data/listing', fn)
            df = pd.read_csv(_fn) 
            df.to_sql('listing', con=engine, if_exists='replace')

    for fn in os.listdir('/home/ubuntu/s3data/average-price'):
        if ".csv" in fn:
            _fn = os.path.join('/home/ubuntu/s3data/average-price', fn)
            df = pd.read_csv(_fn) 
            df.to_sql('average', con=engine, if_exists='replace')


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route("/hosts")
@cross_origin()
def get_hosts():
    query = "SELECT DISTINCT host_id FROM listing GROUP BY host_id HAVING count(host_id) >= 5 ORDER BY RANDOM() limit 5"
    result = engine.execute(query).fetchall()

    return jsonify({'result': [dict(row) for row in result]})

@app.route("/query")
@cross_origin()
def query():
    host_id = request.args.get('host-id', type = int)
    _id = request.args.get('id', type = int)
    if not host_id and not _id:
        abort(400)

    if host_id:
        app.logger.info("host id: %s is querying", host_id)
        query = "SELECT * FROM listing WHERE host_id == %s" % host_id
    if _id:
        app.logger.info("id: %s is querying", _id)
        query = "SELECT * FROM listing WHERE id == %s" % _id

    result = engine.execute(query).fetchall()
    query = "SELECT * FROM average"
    result_avg = engine.execute(query).fetchall()

    if not result:
        abort(404)
    listing, clusters = ([dict(row) for row in result], [dict(row) for row in result_avg])

    resp = list()
    for item in listing:
        prediction = item["prediction"]
        app.logger.debug("cluster for item : %s", prediction)

        for cluster in clusters:
            if cluster["prediction"] == prediction:
                item["avg_price"] = cluster["average_price"]
                item["avg_price"] += ts_increase
                resp.append(item)

    return jsonify({'result': resp})

if __name__ == "__main__":
    load_sqlite()
    app.run(host='0.0.0.0', debug=True)
