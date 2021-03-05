from flask import Flask, jsonify, request
import model_loader
import get_predict
import pickle

__version__ = '0.01'

# get version
with open('version.pkl', 'rb') as f:
    model_version = pickle.load(f)

# load model
model = model_loader.load_model()

with open('data/text.pkl', 'rb') as f:
    TEXT = pickle.load(f)

# app
app = Flask(__name__)
# routes
@app.route('/', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    feedback = data['text']
    pred = float(get_predict.interpret_sentence(model, feedback, TEXT))

    func = lambda x: 'positive feedback' if x > 0.5 else 'negative feedback'
    result = {"text": feedback, "score": pred, "decision": func(pred)}

    # return data
    return jsonify(results=result)

@app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        return 'health status = OK'


@app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': model_version['model_version'],
                        'api_version': __version__})

if __name__ == '__main__':
    app.run()
