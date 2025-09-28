# Updated app.py
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json, numpy as np, time
from itertools import combinations
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'HOU@2025'
socketio = SocketIO(app)

'''================================================='''
# Load descriptors
def load_descriptors(DESCRIPTOR_FILE):
    try:
        with open(DESCRIPTOR_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading descriptors: {e}")
        return {}
# Compute distances between persons and between descriptors in persons
def compute_pairwise_distances(descriptors, sfilter):
    user_ids = [k for k in list(descriptors.keys()) if np.any([k.startswith(sf) for sf in sfilter])]
    pairs_min_distances = []
    same_max_distances = []
    # Convert descriptors to NumPy arrays for faster computation
    descriptor_arrays = {uid: np.array(desc) for uid, desc in descriptors.items()}
    # Compute pairwise minimum distances between different users
    print('\nComputing distances between persons...',end='',flush=True)
    k = 0
    for i, j in combinations(range(len(user_ids)), 2):
        user1, user2 = user_ids[i], user_ids[j]
        desc1, desc2 = descriptor_arrays[user1], descriptor_arrays[user2]
        # Compute all pairwise distances between desc1 and desc2 - Shape: (len(desc1), len(desc2))
        distances = np.linalg.norm(desc1[:, np.newaxis] - desc2, axis=2)
        min_distance = np.min(distances)
        pairs_min_distances.append(min_distance)
        k += 1
        if k%10000==0: print(k,end=',',flush=True)
    # Compute maximum distances within the same user
    print('\nComputing distances in same persones...',end='',flush=True)
    k = 0
    for uid in user_ids:
        desc = descriptor_arrays[uid]
        if len(desc) < 2:  # If fewer than 2 descriptors, max distance is undefined
            max_distance = float('-inf')
        else:
            # Compute all pairwise distances within desc - Use triu_indices to get upper triangle indices (exclude self-pairs)
            i, j = np.triu_indices(len(desc), k=1)  # k=1 skips diagonal
            distances = np.linalg.norm(desc[i] - desc[j], axis=1)
            max_distance = np.max(distances) if distances.size > 0 else float('-inf')
        same_max_distances.append(max_distance)
        k += 1
        if k%500==0: print(k,end=',',flush=True)
    print('\nComputed distances with length:',len(pairs_min_distances),'from persons:',len(same_max_distances),flush=True)
    return {'pairs': pairs_min_distances, 'same': same_max_distances, 'filter': sfilter}
# Compute metrics from histories for user
def get_usermetrics(class_id, user_id):
    if class_id in HISTORIES and user_id in HISTORIES[class_id]:
        user_data = HISTORIES[class_id][user_id]
        true_count = user_data.get('true_count', 0)
        neg_count = user_data.get('neg_count', 0)
        total_count = user_data.get('total_count', 0)
        true_pct = round((true_count / total_count * 100) if total_count else 0,2)
        neg_pct = round((neg_count / total_count * 100) if total_count else 0,2)
        fullhist = user_data.get('full_history')
        total_minutes = round((fullhist[-1]['timestamp']-fullhist[0]['timestamp'])/60,2)
        continuous_false = user_data.get('continuous_false_count', 0)
        status = user_data.get('status', 'none')
        return { 'class_id': class_id, 'user_id': user_id, 'match_percentage': true_pct, 'neg_percentage': neg_pct, 
                 'total_minutes':total_minutes, 'continuous_false_count': continuous_false, 'status': status,
        }
    else: return None
'''================================================='''
'''================================================='''
DESCRIPTOR_FILE = 'descriptors.json'
CLASSROOMS = {'0':['duongthanglong','22A1001D0001','22A1001D0345']}
HISTORIES = {}  # {class_id: {user_id: {true_count, neg_count, total_count, continuous_false_count, status, full_history}}}
DESCRIPTORS = load_descriptors(DESCRIPTOR_FILE)
DISTANCES = {} #compute_pairwise_distances(DESCRIPTORS)
SAMETHRESHOLD = 0.7 # Distance descriptors between logged user_id vs current face
FALSETHRESHOLD = 30  # Continuous counting FALSE match in face-recognition (about 5 fps => continuing FALSE in 6 seconds)
INACTIVITY_TIMEOUT = 5  # Seconds to consider a client inactive
'''================================================='''
'''================================================='''
# Background task to check for inactive users in HISTORIES
def check_inactive_users():
    while True:
        current_time = time.time() # ms
        for class_id in HISTORIES:
            for user_id in HISTORIES[class_id]:
                user_data = HISTORIES[class_id][user_id]
                if user_data['full_history']:
                    last_timestamp = user_data['full_history'][-1]['timestamp']
                    if current_time - last_timestamp > INACTIVITY_TIMEOUT and user_data['status']!='stop':
                        user_data.update({'status':'over'})
                        socketio.emit('update_metrics', get_usermetrics(class_id,user_id))
        socketio.sleep(1)
        # time.sleep(1)  # Check every second
# Start background task
socketio.start_background_task(check_inactive_users)
# threading.Thread(target=check_inactive_users, daemon=True).start()
'''================================================='''
@app.route('/')
def index():
    return render_template('index.html', samethreshold=SAMETHRESHOLD, falsethreshold=FALSETHRESHOLD)

@app.route('/monitor')
def monitor():
    return render_template('monitor.html',falsethreshold=FALSETHRESHOLD)

@app.route('/update_faces') #using with '?filter=22A10' to view firstly
def update_faces():
    sfilter = request.args.get('filter')
    if sfilter:
        sfilter = sfilter.split(',')
        print('Filtering=', sfilter, flush=True)
        DISTANCES = compute_pairwise_distances(DESCRIPTORS,sfilter)
        return render_template('update_faces.html', descriptors={}, distances=DISTANCES, samethreshold=SAMETHRESHOLD)
    else:
        return render_template('update_faces.html', descriptors={}, distances={}, samethreshold=SAMETHRESHOLD)

@app.route('/update_descriptors', methods=['POST'])
def update_descriptors():
    data = request.json
    security_code = data.get('security_code')
    if security_code != app.config['SECRET_KEY']:
        return jsonify({'success': False, 'message': 'Invalid security code'}), 403

    new_descriptors = data.get('descriptors')  # dict of {user_id: descriptor_list}
    atIndex = data.get('atIndex',-1)
    print('Got update users:',len(new_descriptors),f'with atIndex={atIndex}',flush=True)
    if not isinstance(new_descriptors, dict):
        return jsonify({'success': False, 'message': 'Invalid descriptors format'}), 400

    # Update descriptors: check if new descriptor is too similar to existing ones
    for user_id, new_descriptor in new_descriptors.items():
        if atIndex<0 or user_id not in DESCRIPTORS:
            DESCRIPTORS[user_id] = new_descriptor #just add/replace only new one
        else:
            if atIndex<len(DESCRIPTORS[user_id]): DESCRIPTORS[user_id][atIndex] = new_descriptor[0] #replacing atIndex
            else: DESCRIPTORS[user_id].extend(new_descriptor)
    try:
        with open(DESCRIPTOR_FILE, 'w') as f:
            json.dump(DESCRIPTORS, f)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error saving descriptors: {e}",flush=True)
        return jsonify({'success': False}), 500

@socketio.on('check_user_id')
def handle_check_user_id(data):
    class_id = data.get('class_id')
    user_id = data.get('user_id')
    if class_id in CLASSROOMS and user_id in CLASSROOMS[class_id] and user_id in DESCRIPTORS:
        # Send to the requested client user: exists
        # print(f'Requested OK new user_id {user_id} with descriptors: {DESCRIPTORS[user_id]}',flush=True)
        emit('user_exists', {'exists': True, 'descriptor': DESCRIPTORS[user_id], 'falseThreshold': FALSETHRESHOLD})
        # Notify to monitor new connected user: all monitor clients        
        if class_id not in HISTORIES: HISTORIES[class_id] = {}
        if user_id not in HISTORIES[class_id]: 
            HISTORIES[class_id][user_id] = {
                'true_count': 0,
                'neg_count': 0,
                'total_count': 0,
                'continuous_false_count': 0,
                'status':'active',
                'full_history': []
            }
        HISTORIES[class_id][user_id].update({'status':'active'})
        # Send metrics to monitor
        socketio.emit('update_metrics', get_usermetrics(class_id,user_id))
    else:
        # Send to the requested client user: no exists!
        emit('user_exists', {'exists': False})

@socketio.on('send_match')
def handle_send_match(data):
    class_id = data.get('class_id')
    user_id = data.get('user_id')
    match = data.get('match')
    fer = data.get('fer')
    timestamp = time.time()  # second

    if class_id not in HISTORIES:
        HISTORIES[class_id] = {}
    if user_id not in HISTORIES[class_id]:
        HISTORIES[class_id][user_id] = {
            'true_count': 0,
            'neg_count': 0,
            'total_count': 0,
            'continuous_false_count': 0,
            'status':'active',
            'full_history': []
        }
    HISTORIES[class_id][user_id].update({'status':'active'})
    user_data = HISTORIES[class_id][user_id]
    user_data['total_count'] += 1
    user_data['neg_count'] += (1 if fer=='Neg' else 0)
    if match:
        user_data['true_count'] += 1        
        user_data['continuous_false_count'] = 0  # Reset on True
    else:
        user_data['continuous_false_count'] += 1  # Increment on False
    user_data['full_history'].append({'timestamp': timestamp, 'match': match, 'fer': fer})    
    
    # Send metrics to monitor
    socketio.emit('update_metrics', get_usermetrics(class_id,user_id))

@socketio.on('user_stop')
def handle_user_stop(data):
    class_id = data.get('class_id')
    user_id = data.get('user_id')
    if class_id in HISTORIES and user_id in HISTORIES[class_id]:
        HISTORIES[class_id][user_id].update({'status':'stop'})
        socketio.emit('update_metrics', get_usermetrics(class_id,user_id))
    else:
        print(f"Invalid stopping an CLASS_ID/USER_ID not in HISTORIES: {class_id}/{user_id}",flush=True)

@socketio.on('update_threshold')
def handle_update_threshold(data):
    global FALSETHRESHOLD
    try:
        new_threshold = int(data.get('falseThreshold'))
        if new_threshold > 0:
            FALSETHRESHOLD = new_threshold
            print(f"Updated falseThreshold to {FALSETHRESHOLD}",flush=True)
            # Broadcast new threshold to all user clients
            socketio.emit('update_threshold', {'falseThreshold': FALSETHRESHOLD})
        else:
            print(f"Invalid threshold: {new_threshold}",flush=True)
    except (ValueError, TypeError) as e:
        print(f"Error updating threshold: {e}",flush=True)

@socketio.on('join_monitor')
def handle_join_monitor():
    # Send metrics for all users and current threshold
    metrics = []
    for class_id in HISTORIES:
        for user_id in HISTORIES[class_id]:
            metrics.append(get_usermetrics(class_id,user_id))
    emit('all_metrics', {'metrics': metrics, 'falseThreshold': FALSETHRESHOLD})

if __name__ == '__main__':
    socketio.run(app, debug=True)