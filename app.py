import os
import secrets
from functools import wraps
from datetime import datetime, time, timedelta
import random

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import desc, func

# AI model imports (assuming waste_classifier.h5 is available)
try:
    import tensorflow as tf
    import numpy as np
    from PIL import Image
    model = tf.keras.models.load_model("waste_classifier.h5")
    labels = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]
    print("✅ AI model loaded successfully.")
except ImportError:
    print("⚠️ TensorFlow or PIL not installed. AI features will be disabled.")
    model = None
except Exception as e:
    print(f"⚠️ Error loading AI model: {e}")
    model = None
    labels = []

# ===============================================
# 1. FLASK APP SETUP
# ===============================================

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Upload folders
UPLOAD_FOLDER = 'static/uploads'
UPLOAD_FOLDER_PICS = 'static/profile_pics'
UPLOAD_FOLDER_TEMP = 'static/temp_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_FOLDER_PICS'] = UPLOAD_FOLDER_PICS
app.config['UPLOAD_FOLDER_TEMP'] = UPLOAD_FOLDER_TEMP

# Create folders if they don't exist
for folder in [UPLOAD_FOLDER, UPLOAD_FOLDER_PICS, UPLOAD_FOLDER_TEMP]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Allowed file types for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Database and Login Manager setup
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ===============================================
# 2. DATABASE MODELS
# ===============================================

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(10), default='user')
    points = db.Column(db.Integer, default=0)
    achievements = db.Column(db.String(200), default='')
    profile_picture = db.Column(db.String(100), nullable=True, default='default.png')
    reports = db.relationship('Report', backref='author', lazy=True)
    school_id = db.Column(db.Integer, db.ForeignKey('school.id'), nullable=True)
    challenges_joined = db.relationship('UserToChallenge', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    report_type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(250), nullable=False)
    status = db.Column(db.String(20), default='pending')
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class School(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    users = db.relationship('User', backref='school', lazy=True)
    bins = db.relationship('Bin', backref='school', lazy=True)

class Bin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    bin_id = db.Column(db.String(50), unique=True, nullable=False)
    location = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(50), default='ok')
    school_id = db.Column(db.Integer, db.ForeignKey('school.id'), nullable=True)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    author = db.relationship('User', backref=db.backref('messages', lazy=True))

class Challenge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    points_reward = db.Column(db.Integer, default=0)
    start_date = db.Column(db.DateTime, default=datetime.utcnow)
    end_date = db.Column(db.DateTime, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    participants = db.relationship('UserToChallenge', backref='challenge', lazy=True)

class UserToChallenge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    challenge_id = db.Column(db.Integer, db.ForeignKey('challenge.id'), nullable=False)
    joined_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_completed = db.Column(db.Boolean, default=False)

# ===============================================
# 3. ROUTES
# ===============================================

# Home page with AI classification
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if 'file' not in request.files:
            flash("No file part", "danger")
            return redirect(url_for('home'))

        file = request.files["file"]
        if file.filename == '':
            flash("No selected file", "danger")
            return redirect(url_for('home'))

        if file and allowed_file(file.filename):
            filename = secrets.token_hex(8) + '.' + file.filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER_TEMP'], filename)
            file.save(filepath)

            if model:
                img = Image.open(filepath).resize((128, 128))
                img = np.array(img) / 255.0
                img = np.expand_dims(img, axis=0)
                pred = model.predict(img)
                predicted_class = labels[np.argmax(pred)]
                os.remove(filepath)
                flash(f"AI classified your item as '{predicted_class}'. Please log in to submit a report.", "success")
            else:
                flash("AI model is not available. Please try again later.", "danger")
        return redirect(url_for('login'))
    return render_template("index.html")

# Authentication Routes
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        user_exists = User.query.filter_by(username=username).first()
        if user_exists:
            flash("Username already exists. Please choose a different one.", "danger")
            return redirect(url_for('register'))
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('dashboard'))
        else:
            flash("Login failed. Check your email and password.", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# Main user dashboard
@app.route("/dashboard")
@login_required
def dashboard():
    active_challenges = Challenge.query.filter(Challenge.end_date >= datetime.utcnow()).all()
    school_leaderboard = []
    school_bins = []
    if current_user.school_id:
        school_leaderboard = User.query.filter_by(school_id=current_user.school_id).order_by(desc(User.points)).all()
        school_bins = Bin.query.filter_by(school_id=current_user.school_id).all()
    else:
        school_leaderboard = User.query.order_by(desc(User.points)).all()
    
    user_challenges = UserToChallenge.query.filter_by(user_id=current_user.id).all()
    user_challenges_data = []
    for uc in user_challenges:
        challenge = db.session.get(Challenge, uc.challenge_id)
        if challenge:
            user_challenges_data.append({'challenge': challenge, 'is_completed': uc.is_completed})
    
    return render_template(
        "dashboard.html",
        user=current_user,
        active_challenges=active_challenges,
        school_leaderboard=school_leaderboard,
        school_bins=school_bins,
        user_challenges=user_challenges_data
    )

# Profile (self)
@app.route("/profile")
@login_required
def profile():
    user_reports = Report.query.filter_by(user_id=current_user.id).order_by(desc(Report.timestamp)).all()
    unlocked_achievements = current_user.achievements.split(',') if current_user.achievements else []
    
    # Dictionary of achievement descriptions for display
    achievement_descriptions = {
        'first_approved_report': 'First Approved Report 🎉',
        'first_challenge_completed': 'Challenge Champion 🏆',
        '100_points': 'Point Pro! 💯',
        '500_points': 'Waste Warrior! 🚀',
        '1000_points': 'Eco Legend! ✨'
    }

    return render_template("profile.html", 
        user=current_user, 
        reports=user_reports, 
        achievements=unlocked_achievements,
        achievement_descriptions=achievement_descriptions
    )

# View another user's profile
@app.route("/user/<int:user_id>")
@login_required
def user_profile(user_id):
    user_to_view = User.query.get_or_404(user_id)
    user_reports = Report.query.filter_by(user_id=user_to_view.id, status='approved').all()
    unlocked_achievements = user_to_view.achievements.split(',') if user_to_view.achievements else []
    
    achievement_descriptions = {
        'first_approved_report': 'First Approved Report 🎉',
        'first_challenge_completed': 'Challenge Champion 🏆',
        '100_points': 'Point Pro! 💯',
        '500_points': 'Waste Warrior! 🚀',
        '1000_points': 'Eco Legend! ✨'
    }

    return render_template("user_profile.html", 
        user=user_to_view, 
        reports=user_reports, 
        achievements=unlocked_achievements,
        achievement_descriptions=achievement_descriptions
    )

# Admin dashboard
@app.route("/admin_dashboard")
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash("You do not have permission to access this page.", "danger")
        return redirect(url_for('dashboard'))
    reports = Report.query.filter_by(status='pending').order_by(desc(Report.timestamp)).all()
    schools = School.query.all()
    users = User.query.all()
    return render_template("admin_dashboard.html", user=current_user, reports=reports, schools=schools, users=users)

# Approve a report
@app.route("/approve_report/<int:report_id>", methods=["POST"])
@login_required
def approve_report(report_id):
    if current_user.role != 'admin':
        flash("You do not have permission to do that.", "danger")
        return redirect(url_for('dashboard'))
    report = Report.query.get_or_404(report_id)
    report.status = 'approved'
    report.author.points += 50
    if report.report_type == 'ai_classified':
        report.author.points += 5
    
    achievements_list = report.author.achievements.split(',') if report.author.achievements else []
    
    if 'first_approved_report' not in achievements_list:
        achievements_list.append('first_approved_report')
    
    if report.author.points >= 100 and '100_points' not in achievements_list:
        achievements_list.append('100_points')
    if report.author.points >= 500 and '500_points' not in achievements_list:
        achievements_list.append('500_points')
    if report.author.points >= 1000 and '1000_points' not in achievements_list:
        achievements_list.append('1000_points')

    report.author.achievements = ','.join(achievements_list)
    
    db.session.commit()
    flash("Report approved and points rewarded!", "success")
    return redirect(url_for('admin_dashboard'))

# Refuse a report
@app.route("/refuse_report/<int:report_id>", methods=["POST"])
@login_required
def refuse_report(report_id):
    if current_user.role != 'admin':
        flash("You do not have permission to do that.", "danger")
        return redirect(url_for('dashboard'))
    report = Report.query.get_or_404(report_id)
    report.status = 'refused'
    db.session.commit()
    flash("Report has been refused.", "info")
    return redirect(url_for('admin_dashboard'))

# Global leaderboard
@app.route("/leaderboard")
def leaderboard():
    top_users = User.query.order_by(desc(User.points)).all()
    return render_template("leaderboard.html", top_users=top_users)

# Manage schools (admin)
@app.route("/manage_schools", methods=['GET', 'POST'])
@login_required
def manage_schools():
    if current_user.role != 'admin':
        flash("You do not have permission to access this page.", "danger")
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        school_name = request.form.get('school_name')
        if school_name:
            new_school = School(name=school_name)
            db.session.add(new_school)
            db.session.commit()
            flash(f"School '{school_name}' added successfully!", "success")
        else:
            flash("School name cannot be empty.", "danger")
        return redirect(url_for('manage_schools'))
    schools = School.query.all()
    return render_template('manage_schools.html', schools=schools)

# Delete school (admin)
@app.route("/delete_school/<int:school_id>", methods=['POST'])
@login_required
def delete_school(school_id):
    if current_user.role != 'admin':
        flash("You do not have permission to do that.", "danger")
        return redirect(url_for('dashboard'))
    school_to_delete = School.query.get_or_404(school_id)
    db.session.delete(school_to_delete)
    db.session.commit()
    flash("School deleted successfully!", "success")
    return redirect(url_for('manage_schools'))

# Bin management for both users and admins
@app.route("/manage_bins", methods=['GET', 'POST'])
@login_required
def manage_bins():
    if request.method == 'POST':
        # Only admins can create new bins
        if current_user.role == 'admin':
            bin_id = request.form.get('bin_id')
            location = request.form.get('location')
            school_id = request.form.get('school_id')
            existing_bin = Bin.query.filter_by(bin_id=bin_id).first()
            if existing_bin:
                flash(f"Bin ID '{bin_id}' already exists. Please use a unique ID.", "danger")
            else:
                new_bin = Bin(bin_id=bin_id, location=location, school_id=school_id if school_id else None)
                db.session.add(new_bin)
                db.session.commit()
                flash(f"Bin '{bin_id}' added successfully!", "success")
            return redirect(url_for('manage_bins'))
        else:
            flash("You do not have permission to add new bins.", "danger")

    # Admins see all bins, users see only their school's bins
    if current_user.role == 'admin':
        bins = Bin.query.all()
        schools = School.query.all()
    else:
        bins = Bin.query.filter_by(school_id=current_user.school_id).all()
        schools = [] # Users don't need to see the school list
    
    return render_template('manage_bins.html', bins=bins, schools=schools, is_admin=(current_user.role == 'admin'))


# Delete bin (admin)
@app.route("/delete_bin/<int:bin_id>", methods=['POST'])
@login_required
def delete_bin(bin_id):
    if current_user.role != 'admin':
        flash("You do not have permission to do that.", "danger")
        return redirect(url_for('dashboard'))
    bin_to_delete = Bin.query.get_or_404(bin_id)
    db.session.delete(bin_to_delete)
    db.session.commit()
    flash("Bin deleted successfully!", "success")
    return redirect(url_for('manage_bins'))

# Bin status (admin) - This route is redundant, manage_bins now handles status updates.
@app.route("/bin_status")
@login_required
def bin_status():
    if current_user.role != 'admin':
        flash("You do not have permission to access this page.", "danger")
        return redirect(url_for('dashboard'))
    bins = Bin.query.all()
    return render_template("bin_status.html", bins=bins)

# Update bin status (both)
@app.route("/update_bin_status/<int:bin_id>", methods=["POST"])
@login_required
def update_bin_status(bin_id):
    bin_to_update = Bin.query.get_or_404(bin_id)
    # Check if user is admin or if the bin belongs to their school
    if current_user.role != 'admin' and bin_to_update.school_id != current_user.school_id:
        flash("You do not have permission to update this bin.", "danger")
        return redirect(url_for('dashboard'))

    new_status = request.form.get("status")
    bin_to_update.status = new_status
    db.session.commit()
    flash(f"Status for Bin '{bin_to_update.bin_id}' updated to '{new_status}'.", "success")
    return redirect(url_for('manage_bins'))

# Stats (admin)
@app.route("/stats")
@login_required
def stats():
    if current_user.role != 'admin':
        flash("You do not have permission to access this page.", "danger")
        return redirect(url_for('dashboard'))
    total_users = User.query.count()
    total_reports = Report.query.count()
    report_type_counts = db.session.query(
        Report.report_type, func.count(Report.id)
    ).group_by(Report.report_type).all()
    report_counts_dict = dict(report_type_counts)
    
    reports_per_school = db.session.query(
        School.name, func.count(Report.id)
    ).join(User, School.id == User.school_id).join(Report, User.id == Report.user_id).group_by(School.name).all()

    report_timeline = db.session.query(
        func.strftime('%Y-%m-%d', Report.timestamp),
        func.count(Report.id)
    ).group_by(func.strftime('%Y-%m-%d', Report.timestamp)).order_by(desc(func.strftime('%Y-%m-%d', Report.timestamp))).limit(7).all()

    return render_template(
        "stats.html",
        total_users=total_users,
        total_reports=total_reports,
        crime_reports=report_counts_dict.get('crime', 0),
        voluntary_reports=report_counts_dict.get('voluntary', 0),
        reports_per_school=reports_per_school,
        report_timeline=report_timeline
    )

# Messages
@app.route("/messages", methods=['GET', 'POST'])
@login_required
def messages():
    if request.method == 'POST':
        message_text = request.form.get('message_text')
        if message_text:
            new_message = Message(text=message_text, user_id=current_user.id)
            db.session.add(new_message)
            db.session.commit()
            flash("Message sent successfully!", "success")
        return redirect(url_for('messages'))
    all_messages = Message.query.order_by(desc(Message.timestamp)).all()
    return render_template("messages.html", messages=all_messages)

# Upload profile picture
@app.route("/upload_profile_picture", methods=['GET', 'POST'])
@login_required
def upload_profile_picture():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No selected file", "danger")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            if current_user.profile_picture and current_user.profile_picture != 'default.png':
                old_pic_path = os.path.join(app.config['UPLOAD_FOLDER_PICS'], current_user.profile_picture)
                if os.path.exists(old_pic_path):
                    os.remove(old_pic_path)
            filename = secrets.token_hex(8) + '.' + file.filename.rsplit('.', 1)[1].lower()
            file_path = os.path.join(app.config['UPLOAD_FOLDER_PICS'], filename)
            file.save(file_path)
            current_user.profile_picture = filename
            db.session.commit()
            flash("Profile picture uploaded successfully!", "success")
            return redirect(url_for('profile'))
        else:
            flash("Allowed file types are png, jpg, jpeg, gif", "danger")
            return redirect(request.url)
    return render_template("upload_profile_picture.html")

# Manage challenges (admin)
@app.route("/manage_challenges")
@login_required
def manage_challenges():
    if current_user.role != 'admin':
        flash("You do not have permission to access this page.", "danger")
        return redirect(url_for('dashboard'))
    challenges = Challenge.query.all()
    return render_template("manage_challenges.html", challenges=challenges)

# Create challenge (admin)
@app.route("/create_challenge", methods=["GET", "POST"])
@login_required
def create_challenge():
    if current_user.role != 'admin':
        flash("You do not have permission to access this page.", "danger")
        return redirect(url_for('dashboard'))
    if request.method == "POST":
        title = request.form.get("title")
        description = request.form.get("description")
        points = request.form.get("points_reward")
        end_date_str = request.form.get("end_date")
        try:
            end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            end_date = datetime.combine(end_date_obj, time(23, 59, 59))
            new_challenge = Challenge(
                title=title,
                description=description,
                points_reward=int(points),
                end_date=end_date
            )
            db.session.add(new_challenge)
            db.session.commit()
            flash("Challenge created successfully!", "success")
            return redirect(url_for('manage_challenges'))
        except (ValueError, TypeError):
            flash("Invalid date or points value.", "danger")
    return render_template("create_challenge.html")

# Manage assignments (admin)
@app.route("/manage_assignments")
@login_required
def manage_assignments():
    if current_user.role != 'admin':
        flash("You do not have permission to do that.", "danger")
        return redirect(url_for('dashboard'))
    unassigned_users = User.query.filter_by(school_id=None, role='user').all()
    unassigned_bins = Bin.query.filter_by(school_id=None).all()
    schools = School.query.all()
    return render_template(
        "manage_assignments.html",
        unassigned_users=unassigned_users,
        unassigned_bins=unassigned_bins,
        schools=schools
    )

# Assign randomly (admin)
@app.route("/assign_randomly", methods=["POST"])
@login_required
def assign_randomly():
    if current_user.role != 'admin':
        flash("You do not have permission to do that.", "danger")
        return redirect(url_for('dashboard'))
    schools = School.query.all()
    unassigned_users = User.query.filter_by(school_id=None, role='user').all()
    if not schools:
        flash("No schools have been registered yet.", "warning")
        return redirect(url_for('manage_assignments'))
    if not unassigned_users:
        flash("All users are already assigned to a school.", "info")
        return redirect(url_for('manage_assignments'))
    for user in unassigned_users:
        random_school = random.choice(schools)
        user.school_id = random_school.id
    db.session.commit()
    flash(f"Successfully assigned {len(unassigned_users)} users to schools!", "success")
    return redirect(url_for('manage_assignments'))

# Assign user manually (admin)
@app.route("/assign_user_manually", methods=["POST"])
@login_required
def assign_user_manually():
    if current_user.role != 'admin':
        flash("You do not have permission to do that.", "danger")
        return redirect(url_for('dashboard'))
    user_id = request.form.get('user_id')
    school_id = request.form.get('school_id')
    if not user_id or not school_id:
        flash("Please select both a user and a school.", "danger")
        return redirect(url_for('manage_assignments'))
    user = User.query.get(user_id)
    school = School.query.get(school_id)
    if not user or not school:
        flash("Invalid user or school selected.", "danger")
        return redirect(url_for('manage_assignments'))
    user.school_id = school.id
    db.session.commit()
    flash(f"User '{user.username}' has been manually assigned to school '{school.name}'!", "success")
    return redirect(url_for('manage_assignments'))

# Report (voluntary / illegal dumping, with image upload)
@app.route("/report", methods=["GET", "POST"])
@login_required
def report():
    if request.method == "POST":
        report_type = request.form.get("report_type")
        description = request.form.get("description")
        if 'file' not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == '':
            flash("No selected file", "danger")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secrets.token_hex(8) + '.' + file.filename.rsplit('.', 1)[1].lower()
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            new_report = Report(
                report_type=report_type,
                description=description,
                image_url=filename,
                user_id=current_user.id
            )
            db.session.add(new_report)
            db.session.commit()
            flash("Report submitted successfully! It is now pending admin approval.", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Allowed file types are png, jpg, jpeg, gif", "danger")
            return redirect(request.url)
    return render_template("report.html")

# Join challenge
@app.route("/join_challenge/<int:challenge_id>", methods=["POST"])
@login_required
def join_challenge(challenge_id):
    challenge = Challenge.query.get_or_404(challenge_id)
    if UserToChallenge.query.filter_by(user_id=current_user.id, challenge_id=challenge_id).first():
        flash("You have already joined this challenge.", "info")
    else:
        new_entry = UserToChallenge(user_id=current_user.id, challenge_id=challenge_id)
        db.session.add(new_entry)
        db.session.commit()
        flash("You have successfully joined the challenge!", "success")
    return redirect(url_for('dashboard'))

# Complete a challenge
@app.route("/complete_challenge/<int:challenge_id>", methods=["POST"])
@login_required
def complete_challenge(challenge_id):
    user_challenge = UserToChallenge.query.filter_by(user_id=current_user.id, challenge_id=challenge_id).first()
    if not user_challenge:
        flash("You are not participating in this challenge.", "danger")
    elif user_challenge.is_completed:
        flash("You have already completed this challenge.", "info")
    else:
        user_challenge.is_completed = True
        challenge = Challenge.query.get(challenge_id)
        current_user.points += challenge.points_reward
        
        achievements_list = current_user.achievements.split(',') if current_user.achievements else []
        if 'first_challenge_completed' not in achievements_list:
            achievements_list.append('first_challenge_completed')
            current_user.achievements = ','.join(achievements_list)
        
        db.session.commit()
        flash(f"Challenge '{challenge.title}' completed! You earned {challenge.points_reward} points.", "success")
    return redirect(url_for('dashboard'))

# New waste classification feature
@app.route('/classify_waste', methods=['GET', 'POST'])
@login_required
def classify_waste():
    if not model:
        flash('The AI model is not available.', 'danger')
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secrets.token_hex(8) + '.' + file.filename.rsplit('.', 1)[1].lower()
            upload_path = os.path.join(app.config['UPLOAD_FOLDER_TEMP'], filename)
            file.save(upload_path)

            try:
                img = Image.open(upload_path).resize((128, 128))
                img = np.array(img) / 255.0
                img = np.expand_dims(img, axis=0)
                predictions = model.predict(img)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = labels[predicted_class_index]
                
                os.remove(upload_path)
                
                return render_template('classify_waste_results.html', prediction=predicted_class)
            except Exception as e:
                flash(f'An error occurred during classification: {e}', 'danger')
                return redirect(url_for('classify_waste'))
        else:
            flash('Invalid file type. Allowed types are png, jpg, jpeg, gif.', 'danger')
            return redirect(url_for('classify_waste'))
    
    return render_template('classify_waste.html')

# ===============================================
# 4. MAIN ENTRY POINT
# ===============================================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username="admin").first():
            admin_user = User(username="admin", email="admin@example.com", role="admin")
            admin_user.set_password("adminpassword")
            db.session.add(admin_user)
            db.session.commit()
            print("✅ Admin user created: admin/adminpassword")
    app.run(debug=True)
