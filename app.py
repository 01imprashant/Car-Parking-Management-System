from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import re

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///parking_lot.db'
CORS(app, origins=["http://127.0.0.1:8800"])
db = SQLAlchemy(app)

class ParkingSlot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    slot_id = db.Column(db.String(10), unique=True, nullable=False)
    status = db.Column(db.String(10), default='vacant')  # 'occupied' or 'vacant'
    reg_number = db.Column(db.String(15), nullable=True)
    mob = db.Column(db.String(15), nullable=True)


def create_tables():
    db.create_all()
    if ParkingSlot.query.count() == 0:
        initial_slots = []
        zones = ['A', 'B', 'C']
        slots_per_zone = 23
        for zone in zones:
            for i in range(1, slots_per_zone + 1):
                slot_id = f"{zone}-{i}"
                initial_slots.append(ParkingSlot(slot_id=slot_id, status='vacant'))

        # Optionally, fill a few slots with sample data for demonstration
        sample_data = [
            ('A-1', 'occupied', 'UP32 12345', '1234567890'),
            ('A-2', 'occupied', 'UP32 67890', '1234567891'),
            ('B-2', 'occupied', 'UP32 54321', '1234567892'),
            ('C-2', 'occupied', 'UP32 98765', '1234567893'),
        ]
        for slot_id, status, reg_number, mob in sample_data:
            spot = next((s for s in initial_slots if s.slot_id == slot_id), None)
            if spot:
                spot.status = status
                spot.reg_number = reg_number
                spot.mob = mob

        db.session.bulk_save_objects(initial_slots)
        db.session.commit()

@app.route('/')
def index():
    spots = ParkingSlot.query.all()
    vacant_count = ParkingSlot.query.filter_by(status='vacant').count()
    occupied_count = ParkingSlot.query.filter_by(status='occupied').count()
    return render_template('index.html', spots=spots, vacant_count=vacant_count, occupied_count=occupied_count)


@app.route('/book', methods=['POST'])
def book_slot():
    slot_id = request.form.get('slot_id')
    reg_number = request.form.get('reg_number')
    mob = request.form.get('mob')

    if not re.match(r'^[A-Z]{2}\d{2} \d{5}$', reg_number):
        return jsonify({'status': 'error', 'message': 'Invalid registration number format! Use: UP32 12345'}), 400

    if not re.match(r'^\d{10}$', mob):
        return jsonify({'status': 'error', 'message': 'Invalid mobile number! Must be 10 digits.'}), 400

    spot = ParkingSlot.query.filter_by(slot_id=slot_id, status='vacant').first()
    if spot:
        spot.status = 'occupied'
        spot.reg_number = reg_number
        spot.mob = mob
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Slot booked successfully!'})
    
    return jsonify({'status': 'error', 'message': 'Slot is already occupied!'}), 409


@app.route('/unbook', methods=['POST'])
def unbook_slot():
    slot_id = request.form['slot_id']
    spot = ParkingSlot.query.filter_by(slot_id=slot_id, status='occupied').first()
    
    if spot:
        spot.status = 'vacant'
        spot.reg_number = None
        spot.mob = None
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Slot unbooked successfully!'})
    
    return jsonify({'status': 'error', 'message': 'Slot is already vacant!'}), 409


@app.route('/find_car', methods=['POST'])
def find_car():
    reg_number = request.form['reg_number']
    spot = ParkingSlot.query.filter_by(reg_number=reg_number).first()
    if spot:
        return jsonify({'status': 'found', 'slot_id': spot.slot_id, 'mob': spot.mob})
    return jsonify({'status': 'not_found', 'message': 'Vehicle not found!'}), 404


@app.route('/refresh', methods=['GET'])
def refresh_slots():
    spots = ParkingSlot.query.all()
    vacant_count = ParkingSlot.query.filter_by(status='vacant').count()
    occupied_count = ParkingSlot.query.filter_by(status='occupied').count()
    
    return jsonify({
        'spots': [{ 
            'slot_id': spot.slot_id, 
            'status': spot.status, 
            'reg_number': spot.reg_number, 
            'mob': spot.mob 
        } for spot in spots],
        'vacant_count': vacant_count,
        'occupied_count': occupied_count
    })

@app.route('/livecar.html')
def livecar():
    return render_template('livecar.html')

if __name__ == '__main__':
    with app.app_context():
        create_tables()
    app.run(debug=True, port=8800)

    
# spot = ParkingSlot.query.filter_by(slot_id=slot_id, status='vacant').first()
# if spot:
#     print(f"Booking slot: {slot_id}")  # Debug line
#     spot.status = 'occupied'
#     spot.reg_number = reg_number
#     spot.mob = mob
#     db.session.commit()
#     print("Slot booked successfully!")  # Debug line