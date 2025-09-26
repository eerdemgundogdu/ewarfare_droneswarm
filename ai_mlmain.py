import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class SwarmLearningModel(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, output_size=64):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        self.position_predictor = nn.Linear(hidden_size, 3)  # x,y,z
        self.health_predictor = nn.Linear(hidden_size, 1)    # health
        
    def forward(self, x):
        encoded = self.shared_encoder(x)
        position = self.position_predictor(encoded)
        health = torch.sigmoid(self.health_predictor(encoded))
        return position, health

class CollaborativeLearning:
    def __init__(self, num_drones):
        self.models = [SwarmLearningModel() for _ in range(num_drones)]
        self.experience_buffer = deque(maxlen=10000)
        self.shared_knowledge = {}
        
    def update_shared_knowledge(self, drone_id, observations):
        """Update Swarm Info"""
        self.shared_knowledge[drone_id] = {
            'timestamp': time.time(),
            'observations': observations,
            'position': observations['position'],
            'health': observations['health']
        }
    
    def predict_damaged_sensor_data(self, damaged_drone_id):
        """Guess the false censor data of a damaged drone"""
        neighbor_data = []
        for drone_id, data in self.shared_knowledge.items():
            if drone_id != damaged_drone_id:
                neighbor_data.append(data['observations'])
        
        if neighbor_data:
            avg_position = np.mean([d['position'] for d in neighbor_data], axis=0)
            avg_health = np.mean([d['health'] for d in neighbor_data])
            
            return {
                'predicted_position': avg_position,
                'predicted_health': avg_health,
                'confidence': 0.8  
        return None