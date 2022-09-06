from flask import Flask, render_template,  redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, login_required, current_user, logout_user, LoginManager

from werkzeug.security import generate_password_hash, check_password_hash

import flask
import pandas as pd
import joblib
import pickle
import numpy as np


app = Flask(__name__)

ENV = 'prod'

if ENV == 'dev':
    app.debug = True
    app.config['SECRET_KEY'] = 'secret-key-goes-here'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:yourpassword@localhost/HeartCare'
else:
    app.config['SECRET_KEY'] = 'secret-key-goes-here'
    app.debug = False
    app.config['SQLALCHEMY_DATABASE_URI']= "your genaerated PostgreSQL config"

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

class User(UserMixin, db.Model):
    __tablename__ = 'users_table'
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    first_name = db.Column(db.String(1000))
    second_name = db.Column(db.String(1000))
    patient_code = db.Column(db.String(1000))
    user_role = db.Column(db.String(1000))

    def  __init__(self, email, password, first_name, second_name, patient_code, user_role):
        self.email = email
        self.password = password
        self.first_name = first_name
        self.second_name = second_name
        self.patient_code = patient_code
        self.user_role = user_role

@login_manager.user_loader
def load_user(user_id):
    # since the user_id is just the primary key of our user table, use it in the query for the user
    return User.query.get(int(user_id))


with open("model_two.pkl", 'rb') as f:
    my_model = pickle.load(f)

thingspeak_data = pd.read_csv('https://api.thingspeak.com/channels/1685425/feeds/last.csv?api_key=JV9VJZA9OH9ONXBV&results=2')


trestbps = thingspeak_data.field4[0]
chol = thingspeak_data.field8[0]
fbs	= thingspeak_data.field5[0]
thalach = thingspeak_data.field3[0]
temperature = thingspeak_data.field1[0]
timer = thingspeak_data.created_at[0]


@app.route('/')
def index():
    return render_template('started.html')

@app.route('/patient_page', methods=['POST', 'GET'])
@login_required
def patient_page():
    
    user_duty = current_user.user_role
    new_bp = 130

    if request.method == 'GET':

        return render_template('patient_page.html', fat=chol, glucose_level=fbs, body_temp=temperature, heart_rate=thalach, blood_pressure=new_bp, time=timer, user_type=user_duty)

    if request.method == 'POST':
        
        sex = flask.request.form.get('sex')
        age = flask.request.form.get('age')
        cp = flask.request.form.get('cp')
        exang = flask.request.form.get('exang')
        
        Diagnosis_one = "There is an 80% probability that you have a form of Heart Disease, and therefore advise that you visit the nearest healthcare location"
        Diagnosis_two = "There is an 80% probability that you do not have any heart disease."

        if fbs > 120:
            fbs_input = 1
        else:
            fbs_input = 0


        input_variables = pd.DataFrame([[age, sex, cp, new_bp, chol, fbs_input, thalach, exang]], columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'exang'], dtype=int)

        prediction = my_model.predict(input_variables)[0]

        if prediction == 0 :
            return render_template('patient_page.html', working = Diagnosis_two, fat=chol, glucose_level=fbs, body_temp=temperature, heart_rate=thalach, blood_pressure=new_bp, time=timer, user_type=user_duty)
        else:
            return render_template('patient_page.html', working = Diagnosis_one, fat=chol, glucose_level=fbs, body_temp=temperature, heart_rate=thalach, blood_pressure=new_bp, time=timer, user_type=user_duty)



@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    # login code goes here
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    user = User.query.filter_by(email=email).first()

    # check if the user actually exists
    # take the user-supplied password, hash it, and compare it to the hashed password in the database
    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again')
        return redirect(url_for('login')) # if the user doesn't exist or password is wrong, reload the page

    # if the above check passes, then we know the user has the right credentials
    login_user(user, remember=remember)
    return redirect(url_for('patient_page'))


@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/signup', methods=['POST', 'GET'])
def signup_post():
    if request.method == 'POST':
        
        email = request.form.get('email')
        first_name = request.form.get('first_name')
        second_name = request.form.get('second_name')
        password = request.form.get('password')
        user_role = request.form.get('user_role')

        user = User.query.filter_by(email=email).first() # if this returns a user, then the email already exists in database

        if user: # if a user is found, we want to redirect back to signup page so user can try again
            flash('This user entity already exists')
            return redirect(url_for('signup'))

        new_user = User(email=email, first_name=first_name, second_name=second_name, password=generate_password_hash(password, method='sha256'), user_role=user_role, patient_code=0000)

        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    else:
        return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run()
