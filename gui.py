from flask import Flask
from flask import request
from flask import render_template
import re
import linear_learner
import logistic_learner
import bulk_learner
import knn_learner
import plot_only

app = Flask(__name__)


def analyse_stock(learner, ticker, options):
    confidence_score, predicted_result = learner.analyse(ticker, options)
    print(confidence_score)
    return confidence_score, predicted_result


@app.route('/')
def my_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def my_form_post():
    if request.method == 'POST':
        options = []
        ticker = request.form['ticker']
        chosen_learner = request.form.get('chosen_learner')

        # remove all the white space of the input(Ticker)
        ticker = ticker.rstrip()
        ticker = ticker.lstrip()
        ticker = re.sub('[\s+]', '', ticker)

        if not ticker:
            return "ah... Sorry! I don't understand what you mean. Please try again.."

        if request.form.get('10ma'):
            options.append('10ma')
        if request.form.get('100ma'):
            options.append('100ma')
        if request.form.get('250ma'):
            options.append('250ma')
        if request.form.get('candlestick'):
            options.append('candlestick')

        if str(chosen_learner) == 'svr_learner':
            confidence_score, predicted_result = analyse_stock(linear_learner, ticker, options)
            return render_template('linear_result.html', con=confidence_score, prediction=str(predicted_result), ticker_name=ticker)
        elif str(chosen_learner) == 'logistic_learner':
            confidence_score, predicted_result = analyse_stock(logistic_learner, ticker, options)
            return render_template('logis_result.html', con=confidence_score, prediction=str(predicted_result), ticker_name=ticker)
            # return "Confidence Score of Logistic Regression Analysis: " + str(confidence_score) + "Predicted Result: " + str(predicted_result)
        elif str(chosen_learner) == 'knn_learner':
            confidence_score, predicted_result = analyse_stock(knn_learner, ticker, options)
            return render_template('knn_result.html', con=confidence_score, prediction=str(predicted_result), ticker_name=ticker)
        elif str(chosen_learner) == 'bulk_learner':
            confidence_score, predicted_result = analyse_stock(bulk_learner, ticker, options)
            return render_template('bulk_result.html', con_list=confidence_score, prediction=str(predicted_result), ticker_name=ticker)
        elif str(chosen_learner) == 'None' and not ticker == 'None':
            plot_only.draw_line_chart(ticker, options)
            return "You didn't select an option"
        else:
            return "die"
        # return render_template("linear_result.html", result=result)

if __name__ == '__main__':
    app.run()