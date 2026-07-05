# Importing essential libraries
from flask import Flask, render_template, redirect, url_for, flash, request,session
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from keras.models import load_model
import os
import csv
import secrets
import sqlite3
import datetime
import time

model = load_model('cnn.h5')

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
# Function to connect to SQLite database
def connect_db():
    conn = sqlite3.connect('users.db')
    return conn

# Function to create users table
def create_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            mobile TEXT NOT NULL,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            address TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Function to insert a new user into the database
def insert_user(name, email, mobile, username, password, address):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO users (name, email, mobile, username, password, address) 
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, email, mobile, username, password, address))
    conn.commit()
    conn.close()

# Function to authenticate user login
def authenticate_user(username, password):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM users WHERE username = ? AND password = ?
    ''', (username, password))
    user = cursor.fetchone()
    conn.close()
    return user



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        mobile = request.form['mobile']
        username = request.form['username']
        password = request.form['password']
        address = request.form['address']

        insert_user(name, email, mobile, username, password, address)
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['login_username']
        password = request.form['login_password']

        user = authenticate_user(username, password)
        if user:
            session['user_id'] = user[0]
            return redirect(url_for('upload'))
        else:
            return render_template('login.html', message='Invalid username or password.')
    return render_template('login.html')



UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Email configuration
sender_email = "malarvizhi.ramu1986@gmail.com"
receiver_email = "test@gmail.com"
subject = "Spam Message Detected Alert"

# Set up the MIME
message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = subject

# Attach the body to the email

# SMTP server configuration (example for Gmail)
smtp_server = "smtp.gmail.com"
smtp_port = 587
smtp_username = "malarvizhi.ramu1986@gmail.com"
smtp_password = "yazyisjpzfsiqwes"
filename = "model/model.pkl"
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open("model/cv-transform.pkl",'rb'))


@app.route("/adminlogin", methods=["GET", "POST"])
def adminlogin():
    
    if request.method == "POST":
        first_name = request.form.get("username")
        last_name = request.form.get("password") 
        if first_name == 'admin' and last_name == 'admin':
            return redirect("/upload")
    else:
        return render_template("adminlogin.html")

@app.route('/home')
def home():
	return render_template('home.html')

# Define the folder where spam messages are stored
SPAM_FOLDER = "spam_messages"

# Ensure the spam folder exists
if not os.path.exists(SPAM_FOLDER):
    os.makedirs(SPAM_FOLDER)
from apscheduler.schedulers.background import BackgroundScheduler
import atexit


def delete_old_spam_files():
    """Deletes spam message files older than 2 minutes."""
    now = time.time()
    for filename in os.listdir(SPAM_FOLDER):
        file_path = os.path.join(SPAM_FOLDER, filename)
        if os.path.isfile(file_path):
            # Get file creation time (on Windows, this is the creation time; on Unix, it may be the last metadata change)
            creation_time = os.path.getctime(file_path)
            # If file is older than 120 seconds (2 minutes), delete it
            if now - creation_time > 120:
                try:
                    os.remove(file_path)
                    app.logger.info(f"Deleted old spam file: {file_path}")
                except Exception as e:
                    app.logger.error(f"Error deleting file {file_path}: {e}")


# Set up a background scheduler to run the cleanup job every minute
scheduler = BackgroundScheduler()
scheduler.add_job(func=delete_old_spam_files, trigger="interval", seconds=60)
scheduler.start()

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        msg = request.form['message']
        name = request.form['name']
        email = request.form['email']
        mobile = request.form['phone']

        # Vectorize and classify the message
        data = [msg]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)[0]

        # If the message is classified as spam, store it as an individual text file
        if my_prediction == 1:
            # Generate a unique filename based on the current timestamp
            filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".txt"
            file_path = os.path.join(SPAM_FOLDER, filename)
            # Write the spam details into the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Name: {name}\nEmail: {email}\nMobile: {mobile}\nMessage: {msg}\n")
            
            # (Optional) Send an email alert for spam
            body = f"The Username is: {name}\nMobile Number: {mobile}\nUser Message: {msg}"
            message = MIMEText(body, "plain")
            message['Subject'] = "Spam Alert"
            message['From'] = smtp_username
            message['To'] = receiver_email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()  # Secure the connection
                server.login(smtp_username, smtp_password)
                server.sendmail(smtp_username, receiver_email, message.as_bytes())

        return render_template('result.html', prediction=my_prediction)
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview', methods=['POST'])
def preview():
    if request.method == 'POST':
        csv_data = parse_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'spam.csv'))
        return render_template('preview.html', csv_data=csv_data)

def parse_csv(file_path, limit=100):
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        data = []
        for i, row in enumerate(csvreader):
            if i >= limit:
                break
            data.append(row)
        return data
@app.route('/')
def slider():
    return render_template('slider.html')  
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    create_table()
    app.run()