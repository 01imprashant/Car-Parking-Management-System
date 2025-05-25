import cv2
import pickle
import cvzone
import time
import numpy as np
from flask import Flask, render_template, request, jsonify
import base64
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
import re

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///parking_lot.db'
db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow cross-origin requests

# Video feed
cap = cv2.VideoCapture('carPark.mp4')

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

# Add zone labels
zones = ['A', 'B', 'C']
width, height = 107, 48

# Global variable to track occupied slots
occupied_slots = []


# Mouse callback function to select parking slots manually
def mouseClick(events, x, y, flags, params):
    if events == cv2.EVENT_LBUTTONDOWN:  # Left click to add positions
        posList.append((x, y))
    elif events == cv2.EVENT_RBUTTONDOWN:  # Right click to remove positions
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)


# Function to check parking space
def checkParkingSpace(imgProcess, img):
    spaceCounter = 0
    global occupied_slots
    occupied_slots = []  # Reset the list for each frame
    
    zone_count = len(zones)  # Number of zones
    slots_per_zone = 23      # Number of slots per zone (you can change this as needed)
    
    for idx, pos in enumerate(posList):
        x, y = pos
        imgCrop = imgProcess[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)
        
        # Determine the zone and slot number dynamically
        zone_index = idx // slots_per_zone  # Determine which zone
        if zone_index >= zone_count:
            continue  # Skip if zone index is out of bounds
        
        zone_number = zones[zone_index]  # Get the zone letter
        slot_number = (idx % slots_per_zone) + 1  # Slot numbers start from 1
        
        full_slot_number = f"{zone_number}-{slot_number}"

        if count < 900:
            color = (0, 255, 0)  # Green for free
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)  # Red for occupied
            thickness = 2
            occupied_slots.append(full_slot_number)

        # Annotate the slot number and draw rectangle on the image
        # cvzone.putTextRect(img, full_slot_number, (x, y + height - 20), scale=1.5, thickness=2, offset=0)
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        # cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color)

    total_slots = 69
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{total_slots}', (100, 50),
                       scale=3, thickness=5, offset=20, colorR=(0, 200, 0))
    # cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3, thickness=5, offset=20, colorR=(0,200,0))
    
    return occupied_slots, spaceCounter



class ParkingSlot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    slot_id = db.Column(db.String(10), unique=True, nullable=False)
    status = db.Column(db.String(10), default='vacant')  # 'occupied' or 'vacant'
    reg_number = db.Column(db.String(15), nullable=True)
    mob = db.Column(db.String(15), nullable=True)



def sync_db_with_detection(occupied_slots):
    # Get all slots from DB
    all_slots = ParkingSlot.query.all()
    for slot in all_slots:
        if slot.slot_id in occupied_slots:
            if slot.status != 'occupied':
                slot.status = 'occupied'
                slot.reg_number = None  # Optionally clear reg/mob if detected by camera
                slot.mob = None
        else:
            if slot.status != 'vacant':
                slot.status = 'vacant'
                slot.reg_number = None
                slot.mob = None
    db.session.commit()


# Function to update parking status
def update_parking_status():
    while True:
        success, img = cap.read()
        if not success:
            # print("Failed to read frame. Restarting video...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # print("Emitting frame at", time.time())     
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
        imgMedian = cv2.medianBlur(imgThreshold, 5)
        kernel = np.ones((3, 3), np.uint8)
        imgDil = cv2.dilate(imgMedian, kernel, iterations=1)

        occupied_slots, free_count = checkParkingSpace(imgDil, img)

        # --- Sync DB with detection ---
        with app.app_context():
            sync_db_with_detection(occupied_slots)
        # -----------------------------

        is_full = len(occupied_slots) == len(posList)
        
        # Encode the image in base64 format for display on the frontend
        _, buffer = cv2.imencode('.jpg', img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        parking_status = {
            'free_slots': free_count,
            'occupied_slots': occupied_slots,
            'is_full': is_full,
            'image': image_base64
        }
        
        # Emit the parking status to all connected clients
        socketio.emit('update', parking_status)
        time.sleep(0.1)

# Route for the index page
@app.route('/')
def index():
    return render_template('livecar.html')


# Route to find car location
@app.route('/find_car_location', methods=['POST'])
def find_car_location():
    car_slot = request.form.get('slot_number')
    if car_slot in [f"{zone}-{slot}" for zone in zones for slot in range(1, 6)]:
        # Extract the zone and slot number
        zone, slot = car_slot.split('-')
        zone_index = zones.index(zone)
        slot_index = int(slot) - 1  # Convert to zero-indexed
        
        if zone_index * 5 + slot_index < len(posList):
            return jsonify({"slot": car_slot, "position": posList[zone_index * 5 + slot_index]})
    return jsonify({"error": "Invalid slot number."}), 400



# Route to manually add or remove parking positions
@app.route('/edit_positions')
def edit_positions():
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Display all parking spaces
        for pos in posList:
            cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (0, 255, 0), 2)
        
        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", mouseClick)

        # Press 's' to save, 'q' to quit
        key = cv2.waitKey(1)
        if key == ord('s'):  # Save the positions
            with open('CarParkPos', 'wb') as f:
                pickle.dump(posList, f)
            print("Positions saved.")
        elif key == ord('q'):  # Quit the position editor
            break

    cv2.destroyAllWindows()
    return "Positions updated."



# Main function to run the app
if __name__ == "__main__":  # Fixed here
    # Start the parking status update in a separate thread
    socketio.start_background_task(update_parking_status)
    socketio.run(app, debug=True)



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


if __name__ == '__main__':
    with app.app_context():
        create_tables()  # Create tables when the application starts
    socketio.start_background_task(update_parking_status)  # Start background task
    socketio.run(app, debug=True, port=5000)  # Change 5001 to your desired port number
