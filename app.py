from flask import Flask, json, g, request, render_template
from flask_cors import CORS

from lib.ai import *
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

# set the project root directory as the static folder, you can set others.
app = Flask(__name__, static_url_path='')
CORS(app)


@app.route('/')
def root():
    return app.send_static_file('chatbot.html')


@app.route("/api/ask", methods=["POST"])
def index():
    data = json.loads(request.data)
    query = data["query"]
    model = data["model"]

    response = ''
    # if model is DAM
    if model == 'w2v-wmd-dam':
        # if it is the first conversation pair, no contexts, but updates the contexts later
        # else if it isn't the first pair, retrieve contexts for DAM, update the contexts afterwards ready for next pair
        if r.hgetall('prev-utterances') == {}:
            response = process_query(query, model, {})
            prev_dict = {'u1': query, 'u2': response}
            r.hmset('prev-utterances', prev_dict)
        else:
            response = process_query(query, model, r.hgetall('prev-utterances'))
            prev_dict = {'u1': query, 'u2': response}
            r.hmset('prev-utterances', prev_dict)
    # if model is not DAM, it doesn't need contexts; if contexts exist, delete them
    else:
        response = process_query(query, model, {})
        if r.hgetall('prev-utterances') != {}:
            r.delete('prev-utterances')

    return json_response({"message":response})


def json_response(payload, status=200):
    return (json.dumps(payload), status, {'content-type': 'application/json'})


if __name__ == '__main__':
    app.run()
