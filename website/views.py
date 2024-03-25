from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user

from ml_model import Asset
from .models import Note, User
from . import db
import json

Assets = Asset('./NotCropedPhoto/temp.jpg', './CropedPhotoTemp/croped.jpg', 
               "./models/AgeGender_fp16.tflite", "./models/height_weight_models.tflite")

views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    return render_template("home.html", user=current_user)

@views.route('/')
@login_required
def questions():
    return render_template("questions.html", user=current_user)
@views.route('/answer', methods=['POST'])
def answer_questions():
    def determine_active_category(answers):
    # Define the rules and scores for each active category
        categories = {
            'Sedentary': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Assign score 1 to each sedentary answer
            'Lightly active': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # Assign score 2 to each lightly active answer
            'Moderately active': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # Assign score 3 to each moderately active answer
            'Very active': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # Assign score 4 to each very active answer
            'Extra active': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # Assign score 5 to each extra active answer
        }

        # Calculate total score for each active category
        scores = {category: sum(categories[category][i] for i in range(10)) for category in categories}

        # Determine the active category with the highest total score
        active_category = max(scores, key=scores.get)

        return active_category

    # Example usage:
    answers = {
        'q1_answer': 3,  # Answer to question 1
        'q2_answer': 4,  # Answer to question 2
        'q3_answer': 2,  # Answer to question 3
        'q4_answer': 5,  # Answer to question 4
        'q5_answer': 4,  # Answer to question 5
        'q6_answer': 1,  # Answer to question 6
        'q7_answer': 3,  # Answer to question 7
        'q8_answer': 4,  # Answer to question 8
        'q9_answer': 5,  # Answer to question 9
        'q10_answer': 3  # Answer to question 10
    }

    active_category = determine_active_category(answers)
    print("Active category:", active_category)

@views.route('/faceDetect', methods=['POST'])
def faceDetect():
        # getting image data from post request
        file = request.files['image']
        # saving image
        file.save("NotCropedPhoto/temp.jpg")
        # ckeck if face detectable
        faceDetectAble = True
        try:
            Assets.crop_save_image()
        except:
            faceDetectAble = False
        # create result
        result = {
            'PIC_prediction': str(faceDetectAble)
        }
        return jsonify(result)    




@views.route('/getagegender',methods=['POST'])
def getagegender():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    Asset.crop_save_image()
    arr = Assets.img2arr('./CropedPhotoTemp/croped.jpg', 1)

    predictions = Assets.make_AgeGender_predictions(arr)

    result = {
        'PIC_prediction': list(predictions)
    }
    return jsonify(result)


@views.route('/getheightweight',
           methods=['POST'])
def getheightweight():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    Assets.crop_save_image()
    arr = Assets.img2arr('./CropedPhotoTemp/croped.jpg', 1)

    predictions = Assets.make_HeightWeight_predictions(arr)

    result = {
        'PIC_prediction': list(predictions)
    }
    return jsonify(result)


@views.route('/getheightweightagegender',
           methods=['POST'])  # you will get method not allowed in brower because browser sends GET request
def getheightweightagegender():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    Assets.crop_save_image()
    arr = Assets.img2arr('./CropedPhotoTemp/croped.jpg', 1)

    predictionsAG = Assets.make_AgeGender_predictions(arr)
    predictionsHW = Assets.make_HeightWeight_predictions(arr)

    result = {
        'PIC_prediction': list(predictionsAG + predictionsHW)
    }
    return jsonify(result)


@views.route('/getbmr', methods=['POST'])  # you will get method not allowed in brower because browser sends GET request
def getBMR():
    file = request.files['image']
    file.save("NotCropedPhoto/temp.jpg")
    Assets.crop_save_image()
    arr = Assets.img2arr('./CropedPhotoTemp/croped.jpg', 1)

    predictionsAG = Assets.make_AgeGender_predictions(arr)
    predictionsHW = Assets.make_HeightWeight_predictions(arr)
    BMR = Assets.AGHWToBMR(predictionsAG[0], predictionsAG[1], predictionsHW[0], predictionsHW[1])

    result = {
        'PIC_prediction': int(BMR)
    }
    return jsonify(result)


@views.route('/ping', methods=['GET'])
def ping():
    return "Pinging Model is really use full! Yay!!!!!!!!!"


@views.route('/addData', methods=['POST'])
def post_users():
    data = request.data.decode('utf-8')
    jsondata = json.loads(data)
    _username = jsondata['username']
    _BMI = float(jsondata['BMI'])
    _BMR = int(jsondata['BMR'])
    
    # Create a new User object and add it to the database
    new_user = User(username=_username, BMI=_BMI, BMR=_BMR)
    db.session.add(new_user)
    db.session.commit()
    
    return "worked"


@views.route('/getData', methods=['POST'])
def getData():
    data = request.data.decode('utf-8')
    jsondata = json.loads(data)

    _username = jsondata['username']
    
    # Query the database for the user
    user = User.query.filter_by(username=_username).first()
    
    if user:
        result = {
            'BMI_data': user.BMI,
            'BMR_data': user.BMR
        }
    else:
        result = {'error': 'User not found'}
    
    return jsonify(result)



@views.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        f = request.files['file']
        f.save('./NotCropedPhoto/temp.jpg')
        Assets.crop_save_image()
        arr = Assets.img2arr('./CropedPhotoTemp/croped.jpg', 1)
        predictionsAG = Assets.make_AgeGender_predictions(arr)
        predictionsHW = Assets.make_HeightWeight_predictions(arr)
        BMR = Assets.AGHWToBMR(predictionsAG[0], predictionsAG[1], predictionsHW[0], predictionsHW[1])
        print(BMR)
        return render_template("result.html",BMR=int(BMR))


