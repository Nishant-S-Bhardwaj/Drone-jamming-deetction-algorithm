import numpy as np
import matplotlib.pyplot as plt
import hashlib


# Allowed hopping frequencies (in GHz)
hopping_frequencies = [2.4, 2.4212, 2.4334, 2.4112, 2.4621, 2.4628, 2.4751, 2.4322, 2.4830]

# Jammer's frequency pool (in GHz)
jammer_frequency_pool = [2.4, 2.4210, 2.4334, 2.4112, 2.4310, 2.4632, 2.4751, 2.4328, 2.4830]


# Simulate SHA256 hash generation
def sha256_hash(value):
    """
    Simulate a SHA256 hashing mechanism.
    Hash the value passed to it and return the hash string.
    """
    hash_object = hashlib.sha256()
    hash_object.update(value.encode('utf-8'))
    return hash_object.hexdigest()


# Generate chaotic map for frequency hopping sequence with shared initial state
def generate_chaotic_sequence(num_hops, frequencies, initial_state):
    """
    Generate a frequency hopping sequence based on chaotic map principles.
    Ensures synchronization by using the same initial state.
    """
    x = initial_state  # Shared initial state for synchronization
    frequency_sequence = []
    chaotic_a, chaotic_b = 17, 13  # Chaotic map constants
    for i in range(num_hops):
        # Update chaotic map equation
        x = np.sin(2 * np.pi * chaotic_a * x) + np.cos(2 * np.pi * chaotic_b * x)
        # Map chaotic output into the valid frequency list
        index = int(np.mod(x, 1) * len(frequencies))  # Scale to valid indices
        frequency_sequence.append(frequencies[index])
    return frequency_sequence


# Simulate jammer's single active jamming frequency behavior
def generate_jammer_frequencies(num_hops, jammer_frequencies):
    """
    Simulate jammer behavior by dynamically selecting only one frequency to jam at each time step.
    """
    return [np.random.choice(jammer_frequencies) for _ in range(num_hops)]


# Simulate the authentication logic with active jamming
def simulate_authentication(sequence, jammer_frequencies):
    """
    Simulate authentication success/failure during frequency hopping,
    where the jammer blocks a single frequency at each timestep.
    """
    auth_success = []
    R = "0.123456789"  # Fixed random masking value
    for i in range(len(sequence)):
        current_freq = sequence[i]
        # If the current frequency is jammed by the jammer at this timestep, authentication fails
        if np.isclose(current_freq, jammer_frequencies[i], atol=0.001):  # Simulate jamming with tolerance
            auth_success.append(0)  # Authentication failure due to jamming
        else:
            # Normal authentication process
            message_1 = f"{current_freq}-{R}"
            message_2 = f"{current_freq}-{R}"
            # Generate SHA256 hashes
            hash_1 = sha256_hash(message_1)
            hash_2 = sha256_hash(message_2)

            # Compare hashes
            if hash_1 == hash_2:
                auth_success.append(1)  # Authentication success
            else:
                auth_success.append(0)  # Authentication failure
    return auth_success


# Main simulation
num_hops = 20  # Number of hops to simulate
shared_initial_state = 0.5  # Shared initial state value to ensure synchronization

# Step 1: Generate synchronized chaotic sequences for transmitter and receiver
frequency_sequence_transmitter = generate_chaotic_sequence(num_hops, hopping_frequencies, shared_initial_state)
frequency_sequence_receiver = generate_chaotic_sequence(num_hops, hopping_frequencies, shared_initial_state)

# Step 2: Simulate jammer's single active jamming frequency behavior over time
jammer_frequencies = generate_jammer_frequencies(num_hops, jammer_frequency_pool)

# Step 3: Simulate authentication success/failure for both transmitter and receiver under jamming conditions
auth_success_transmitter = simulate_authentication(frequency_sequence_transmitter, jammer_frequencies)
auth_success_receiver = simulate_authentication(frequency_sequence_receiver, jammer_frequencies)

# Step 4: Plotting the results
plt.figure(figsize=(15, 20))

# Plot 1: Transmitter frequency hopping sequence
plt.subplot(5, 1, 1)
plt.plot(range(1, num_hops + 1), frequency_sequence_transmitter, marker='o', linestyle='-', linewidth=2)
plt.xlabel('Time Steps')
plt.ylabel('Transmitter Frequency (GHz)')
plt.title('Transmitter Frequency Hopping Sequence')

# Plot 2: Receiver frequency hopping sequence
plt.subplot(5, 1, 2)
plt.plot(range(1, num_hops + 1), frequency_sequence_receiver, marker='o', linestyle='-', linewidth=2)
plt.xlabel('Time Steps')
plt.ylabel('Receiver Frequency')
plt.title('Receiver Frequency Hopping Sequence')

# Plot 3: Jammer's active jamming frequency over time
plt.subplot(5, 1, 3)
plt.plot(range(1, num_hops + 1), jammer_frequencies, marker='o', linestyle='-', color='r')
plt.xlabel('Time Steps')
plt.ylabel('Jammed Frequency')
plt.title('Jammer\'s Single Active Frequency Over Time')

# Plot 4: Transmitter authentication success over time
plt.subplot(5, 1, 4)
plt.plot(range(1, num_hops + 1), auth_success_transmitter, marker='o', linestyle='-', color='g')
plt.xlabel('Time Steps')
plt.ylabel('Authentication Success')
plt.title('Transmitter Authentication Success')

# Plot 5: Receiver authentication success over time
plt.subplot(5, 1, 5)
plt.plot(range(1, num_hops + 1), auth_success_receiver, marker='o', linestyle='-', color='b')
plt.xlabel('Time Steps')
plt.ylabel('Authentication Success')
plt.title('Receiver Authentication Success')

plt.tight_layout()
plt.show()
