import json
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import numpy as np
#from scipy.signal import butter, filtfilt, correlate, periodogram
import datetime
import pickle
#from sklearn.preprocessing import LabelEncoder
import math

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user_database_cmx4_user:Yv6hGck0QJXBZgWburCYjbk5P9OCqREf@dpg-ch4nj533cv23dkld2id0-a/user_database_cmx4'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 

db = SQLAlchemy(app)
filename = "decision_tree.pickle"

class User(db.Model):
    tablename = 'user_details'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    gender = db.Column(db.String(50))
    age = db.Column(db.Integer)
    weight = db.Column(db.Integer)
    height = db.Column(db.Float)
    bmi = db.Column(db.Float)
    heartrate = db.Column(db.Integer)
    arthype = db.Column(db.String(50))
    subhypo = db.Column(db.String(50))
    dia = db.Column(db.String(50))
    ldopa = db.Column(db.Integer)
    orthohypo = db.Column(db.String(50))
    basetemp = db.Column(db.Float)
    handtemp = db.Column(db.Float)
    thirdfingtemp = db.Column(db.Float)
    rr = db.Column(db.Float)
    
        
    device_id = db.Column(db.String(100))
    
    

    def __init__(self, name,age,device_id,gender,weight,height,ldopa,rr,
                 bmi,basetemp,handtemp,thirdfingertemp,dia,arthype,heartrate,
                 subhypo,orthohypo):
        self.name = name
        self.gender = gender
        self.age = age
        self.weight = weight
        self.height = height
        self.bmi = bmi
        self.heartrate = heartrate
        self.arthype = arthype
        self.subhypo = subhypo
        self.dia = dia
        self.ldopa = ldopa
        self.orthohypo = orthohypo
        self.basetemp = basetemp
        self.handtemp = handtemp
        self.thirdfingertemp = thirdfingertemp
        self.rr = rr
        
        self.device_id = device_id
        
        

                
        
class SensorValues(db.Model):
    tablename = 'sensor_values'
    id = db.Column(db.Integer, primary_key=True)
    #user_id = db.Column(db.Integer, db.ForeignKey('parent_table.id'), nullable=False)
    acc_x = db.Column(db.Float) 
    acc_y = db.Column(db.Float)
    acc_z = db.Column(db.Float)
    gyr_x = db.Column(db.Float) 
    gyr_y = db.Column(db.Float)
    gyr_z = db.Column(db.Float)
    

    #{'sensorType': 'Temperature', 'values': [20, 21, 23], 'timestamps': ['10:10', '10:20', '10:30']}


    def __init__(self, acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z):
        self.acc_x = acc_x
        self.acc_y = acc_y
        self.acc_z = acc_z
        self.gyr_x = gyr_x
        self.gyr_y = gyr_y
        self.gyr_z = gyr_z 

class FreqValues(db.Model):
    tablename = 'freq_values'
    id = db.Column(db.Integer, primary_key=True)
    #user_id = db.Column(db.Integer, db.ForeignKey('parent_table.id'), nullable=False)
    freq = db.Column(db.Float) 
    time_ = db.Column(db.DateTime)
    

    

    def __init__(self, freq, time_):
        self.freq = freq
        self.time_ = time_

@app.route('/')
def index():
    return "Bhuvan Chand"

@app.route('/add',methods=['POST'])
def add():
    add_request = User(name = request.form.get('name'), 
                  age = request.form.get('age'),device_id = request.form.get('device_id'),
                  gender = request.form.get('gender'),ldopa = request.form.get('ldopa'),
                  bmi = request.form.get('bmi'),rr = request.form.get('rr'),
                  basetemp = request.form.get('basetemp'),handtemp = request.form.get('handtemp'),
                  thirdfingertemp = request.form.get('thirdfingertemp'),dia = request.form.get('dia'),
                  height = request.form.get('height'),weight = request.form.get('weight'),
                  heartrate = request.form.get('heartrate'),orthohypo = request.form.get('orthohypo'),
                  subhypo = request.form.get('subhypo'),arthype = request.form.get('arthype'))
    

    user = User.query.filter_by(device_id = request.form.get('device_id')).first()
    if user is None:
        db.session.add(add_request)
        db.session.commit()
        new_user = User.query.filter_by(device_id = request.form.get('device_id')).first()    
        return jsonify({"user":1,"name":new_user.name, "age":new_user.age,
                        "gender":new_user.gender, "ldopa":new_user.ldopa,
                        "bmi":new_user.bmi, "rr":new_user.rr,
                        "basetemp":new_user.basetemp, "thirdfingertemp":new_user.thirdfingertemp,
                        "handtemp":new_user.handtemp, "dia":new_user.dia,
                        "height":new_user.height, "weight":new_user.weight,
                        "heartrate":new_user.heartrate,"orthohypo":new_user.orthohypo,
                        "subhypo":new_user.subhypo,"arthype":new_user.arthype})        
    
    else:
        return jsonify({"user":0})        

@app.route('/edit',methods=['POST'])
def edit():    
   
    new_name = request.form.get('name')
    new_age = request.form.get('age')
    new_gender = request.form.get('gender')
    new_ldopa = request.form.get('ldopa')
    new_bmi = request.form.get('bmi')
    new_rr = request.form.get('rr')
    new_basetemp = request.form.get('basetemp')
    new_handtemp = request.form.get('handtemp')
    new_thirdfingtemp = request.form.get('thirdfingtemp')
    new_dia = request.form.get('dia')
    new_height = request.form.get('height')
    new_weight = request.form.get('weight')
    new_heartrate = request.form.get('heartrate')
    new_orthohypo = request.form.get('orthohypo')
    new_subhypo = request.form.get('subhypo')
    new_arthype = request.form.get('arthype')
    
    user = User.query.filter_by(device_id = request.form.get('device_id')).first()
    if user is None:
        return jsonify({"user":0})
    else:
    
        user.name = new_name
        user.age = new_age
        user.gender = new_gender
        user.ldopa = new_ldopa
        user.bmi = new_bmi
        user.rr = new_rr
        user.basetemp = new_basetemp
        user.thirdfingtemp = new_thirdfingtemp
        user.handtemp = new_handtemp
        user.dia = new_dia
        user.height = new_height
        user.weight = new_weight
        user.heartrate = new_heartrate
        user.orthohypo = new_orthohypo
        user.subhypo = new_subhypo
        user.arthype = new_arthype


        db.session.commit()
    
        return jsonify({"user":1,"name":user.name, "age":user.age,
                        "gender":user.gender, "ldopa":user.ldopa,
                        "bmi":user.bmi, "rr":user.rr, "basetemp":user.basetemp,
                        "thirdfingtemp":user.thirdfingtemp,
                        "handtemp":user.handtemp, "dia":user.dia,
                        "height":user.height, "weight":user.weight, 
                        "heartrate":user.heartrate,"orthohypo":user.orthohypo,
                        "subhypo":user.subhypo,"arthype":user.arthype})

@app.route('/sensor',methods=['POST'])
def sensor():
    content = request.get_json()
    acc_x = content["acc_x"]
    acc_y = content["acc_y"]
    acc_z = content["acc_z"]
    gyr_x = content["gyr_x"]
    gyr_y = content["gyr_y"]
    gyr_z = content["gyr_z"]
    value_request = SensorValues(acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z)
    db.session.add(value_request)
    db.session.commit()
    num_rows = SensorValues.query.count()
    if num_rows >= 1:
        ten_rows = SensorValues.query.order_by(SensorValues.id.desc()).limit(1).all()
        accel_data = np.array([[row.acc_x, row.acc_y, row.acc_z] for row in reversed(ten_rows)])
        gyro_data = np.array([[row.gyr_x, row.gyr_y, row.gyr_z] for row in reversed(ten_rows)])

        freq = find_freq(accel_data,gyro_data)
        print(freq)
        if freq >= 2 and freq <= 100:
            time_ = datetime.datetime.now()
            freq_request = FreqValues(freq, time_)
            db.session.add(freq_request)
            db.session.commit()
        SensorValues.query.delete()
        # SensorValues.query.filter(SensorValues.id.notin_([row.id for row in reversed(ten_rows)])).delete()
        db.session.commit()

    #last_row = SensorValues.query.order_by(SensorValues.id.desc()).limit(10)
    #psd, freq, peaks = process_data(last_row.acc_x, last_row.acc_y, last_row.acc_z, last_row.gyr_x, last_row.gyr_y, last_row.gyr_z)
    #print(psd, freq, peaks)
    #freq = find_freq(last_row.acc_x, last_row.acc_y, last_row.acc_z, last_row.gyr_x, last_row.gyr_y, last_row.gyr_z)


    return jsonify("JSON POSTED")

@app.route('/test',methods=['POST'])
def test():
    freq_table = FreqValues.query.all()
    freq_array = np.array([[row.freq, row.time_] for row in (freq_table)])
    freq_values = freq_array[:,0]
    freq_mean = np.mean(freq_values)
    timestamps = freq_array[:,1]
    freq_array[:,1] = [ts.isoformat() for ts in timestamps]
    
    processed_data = data_preprocessing(freq_mean)
    result_model = mlmodel(processed_data)
    
    result_dict = {"model_result": result_model,
                   "frequency": freq_array[:,0].tolist(), 
                   "time":freq_array[:,1].tolist()}
    
    json_data = json.dumps(result_dict)
    json_data = json_data.replace("\"","'")

    return jsonify(result_dict)

"""
def find_freq(accel_data, gyro_data):
    #accel_data = np.random.normal(size=(1000, 3))
    #gyro_data = np.random.normal(size=(1000, 3))

    accel_data = np.array(accel_data)
    gyro_data = np.array(gyro_data)
    #print(accel_data.shape)

    # Apply high-pass filter to accelerometer data to remove gravity component
    fs = 100  # Sampling rate in Hz
    fc_hp = 1  # Cutoff frequency in Hz for high-pass filter
    b_hp, a_hp = butter(2, fc_hp / (fs/2), btype='highpass')
    accel_data_hp = filtfilt(b_hp, a_hp, accel_data, axis=0)

    # Apply low-pass filter to gyroscope data to remove high-frequency noise
    fc_lp = 10  # Cutoff frequency in Hz for low-pass filter
    b_lp, a_lp = butter(2, fc_lp / (fs/2), btype='lowpass')
    gyro_data_lp = filtfilt(b_lp, a_lp, gyro_data, axis=0)

    # Compute acceleration vector magnitude and angular velocity vector
    accel_mag = np.linalg.norm(accel_data_hp, axis=1)
    gyro_vec = gyro_data_lp

    # Compute cross-correlation between acceleration and angular velocity vectors
    corr = correlate(accel_mag, gyro_vec[:, 0], mode='same')

    # Compute lag between acceleration and angular velocity vectors
    lag = np.argmax(corr) - len(corr)/2

    # Compute power spectral density of acceleration vector
    freq, psd = periodogram(accel_mag, fs=fs, window='hamming', scaling='density')

    # Shift PSD by lag amount
    psd_shifted = np.roll(psd, int(lag))

    # Identify peak frequency in shifted PSD
    tremor_freq = freq[np.argmax(psd_shifted)]

    return tremor_freq"""

def find_freq(accel_data, gyro_data):
    prev_ax, prev_ay, prev_az = 0, 0, 0
    curr_ax, curr_ay, curr_az = 0, 0, 0

    # Define variables for storing the previous and current gyroscope values
    prev_gx, prev_gy, prev_gz = 0, 0, 0
    curr_gx, curr_gy, curr_gz = 0, 0, 0

    # Define variables for storing the displacement values
    dx, dy, dz = 0, 0, 0
    prev_dx, prev_dy, prev_dz = 0, 0, 0

    # Define the time step and start time
    dt = 0.01
    

    # Define the gyroscope drift correction factor
    gyro_corr = 0.98

    # Loop to read and process the accelerometer and gyroscope data
    
        # Read the accelerometer and gyroscope data from the MPU6050
        
        
        # Convert the raw accelerometer data to the correct scale factor
    curr_ax = accel_data[0,0] 
    curr_ay = accel_data[0,1] 
    curr_az = accel_data[0,2] 
        
        # Convert the raw gyroscope data to the correct scale factor and correct for drift
    curr_gx = gyro_data[0,0] 
    curr_gy = gyro_data[0,1] 
    curr_gz = gyro_data[0,2] 
    curr_gx = prev_gx * gyro_corr + curr_gx * (1 - gyro_corr)
    curr_gy = prev_gy * gyro_corr + curr_gy * (1 - gyro_corr)
    curr_gz = prev_gz * gyro_corr + curr_gz * (1 - gyro_corr)
        
        # Calculate the displacement using the trapezoidal integration method
    dx += (prev_ax + curr_ax) * dt / 2
    dy += (prev_ay + curr_ay) * dt / 2
    dz += (prev_az + curr_az) * dt / 2
        
        # Correct for gyroscope drift
    dx -= curr_gx * dt * dt / 2
    dy -= curr_gy * dt * dt / 2
    dz -= curr_gz * dt * dt / 2
        
        # Store the previous acceleration, gyroscope, and displacement values
    prev_ax, prev_ay, prev_az = curr_ax, curr_ay, curr_az
    prev_gx, prev_gy, prev_gz = curr_gx, curr_gy, curr_gz
    prev_dx, prev_dy, prev_dz = dx, dy, dz

        
        
        # Print the displacement values
    #print("dx:", dx, "dy:", dy, "dz:", dz)
    total_displacement = math.sqrt(dx**2 + dy**2 + dz**2)

    total_accel = math.sqrt(curr_ax**2 + curr_ay**2 + curr_az**2)
        
    freq = math.sqrt(total_accel/(2*(3.14**2)*total_displacement))

    return freq


def mlmodel(details_array):
    loaded_model = pickle.load(open(filename, "rb"))
    y_predicted = loaded_model.predict(details_array)
    result_model = y_predicted[0].item()

    return result_model

def data_preprocessing(mean_freq, app_id = "a9c102b71626ec32"):
    user = User.query.filter_by(device_id = app_id ).first()
    gender = user.gender
    age  =user.age
    weight = user.weight
    height = user.height
    bmi = user.bmi
    dia = user.dia
    ldopa = user.ldopa
    basetemp = user.basetemp
    handtemp = user.handtemp
    thirdfingtemp = user.thirdfingtemp
    rr = user.rr
    heartrate = user.heartrate
    orthohypo = user.orthohypo 
    subhypo = user.subhypo  
    arthype = user.arthype 
    

    user_details = np.array([[gender, age, weight, height,bmi, heartrate,
                              arthype, subhypo ,dia,ldopa,mean_freq, orthohypo,
                              basetemp, handtemp,thirdfingtemp, rr]])
    
    
    if user_details[0,0] == 'F':
        user_details[0,0]  = 0
    elif user_details[0,0] =='M':
        user_details[0,0] = 1
    
    if user_details[0,6].lower() =='no':
        user_details[0,6] = 0
    elif user_details[0,6].lower() =='yes':
        user_details[0,6] = 1

    if user_details[0,7].lower() =='no':
        user_details[0,7] = 0
    elif user_details[0,7].lower() =='yes':
        user_details[0,7] = 1

    if user_details[0,8].lower() =='no':
        user_details[0,8] = 0
    elif user_details[0,8].lower() =='yes':
        user_details[0,8] = 1

    if user_details[0,11].lower() =='no':
        user_details[0,11] = 0
    elif user_details[0,11].lower() =='yes':
        user_details[0,11] = 1
    

    return user_details

    
    


    

    
    
if __name__ == '__main__':
    with app.app_context():        
        db.create_all()
    app.run(debug=True, host = "0.0.0.0")