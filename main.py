import platform
from pandas import DataFrame
import matplotlib.pyplot as plt
from tqdm import trange
import os
from simpy import Environment, Resource
from random import sample, expovariate
import numpy as np

class LoadBalancer:
    def __init__(self, env, num_servers, mu, policy='JSQ', d=2):
        self.env : Environment = env
        self.servers = [Server(env, mu, i) for i in range(num_servers)]
        self.policy = policy
        self.d = d  # Number of random choices for JSQd
        self.wait_times = []
        self.queue_stats = []  # Track queue state over time
    
    def choose_server(self):
        if self.policy == 'JSQ':
            return min(self.servers, key=lambda s: s.queue_length)
        elif self.policy == 'JSQd':
            sampled_servers = sample(self.servers, self.d)
            return min(sampled_servers, key=lambda s: s.queue_length)
        else:
            raise ValueError(f"Unknown load balancing policy: {self.policy}")
    
    def dispatch_task(self):
        server = self.choose_server()
        arrival_time = self.env.now
        server.queue_length += 1  # Increment queue length
        
        with server.server.request() as req:
            yield req
            yield self.env.process(server.process_task())
        
        server.queue_length -= 1  # Decrement queue length after service
        self.wait_times.append(self.env.now - arrival_time)
    
    def record_queue_stats(self):
        queue_lengths = np.array([s.queue_length for s in self.servers])
        max_q = queue_lengths.max()
        Q_t = np.bincount(queue_lengths, minlength=max_q + 1)[::-1].cumsum()[::-1]
        self.queue_stats.append((self.env.now, Q_t.tolist()))

class Server:
    def __init__(self, env, mu, id):
        self.id = id
        self.env : Environment = env
        self.server = Resource(env, capacity=1)
        self.mu = mu
        self.queue_length = 0  # Track queue length of this server
    
    def process_task(self):
        service_time = expovariate(self.mu)
        yield self.env.timeout(service_time)

    def __repr__(self):
        return f"Server({self.id})"

def arrival_process(env : Environment, load_balancer : LoadBalancer, lambda_n):
    while True:
        yield env.timeout(expovariate(lambda_n))
        env.process(load_balancer.dispatch_task())
        load_balancer.record_queue_stats()

# Convert queue statistics to DataFrame
def queue_stats_to_df(queue_stats):
    data = []
    longest_queue = max(len(q_vector) for _, q_vector in queue_stats)
    for time, q_vector in queue_stats:
        row = {'time': time}
        row.update({f'Q_{i}': q_vector[i] if i < len(q_vector) else 0 for i in range(1, max(len(q_vector), longest_queue + 1))})
        data.append(row)
    queue_stats_df = DataFrame(data)
    queue_stats_df.insert(1, 'Q_0', 100 - queue_stats_df['Q_1'])
    return queue_stats_df

def plot_queue_stats_over_time(df : DataFrame, title_prefix, ylim = None):
    """
    Plots the queue stats over time for each column in the DataFrame.
    df: DataFrame containing the queue stats with 'time' as the first column.
    title_prefix: Prefix for the plot titles.
    """
    time = df['time']
    for column in df.columns:
        if column == 'time':
            continue
        plt.figure(figsize=(10, 6))
        plt.step(time, df[column], label=column)
        plt.xlabel('Time')
        plt.ylabel(column)
        plt.xlim(0, time.max())
        plt.ylim(0, ylim if ylim else df[column].max() * 1.1)
        plt.title(f'{title_prefix} {column} over Time')
        plt.legend()
        plt.grid()
        plt.savefig(f'{title_prefix}_{column}_over_time.png')
        plt.close()

def plot_wait_time_histogram(wait_times, filename='wait_time_histogram.png'):
    """
    Plots a histogram of wait times.
    wait_times: List of wait times to plot.
    filename: Name of the file to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(wait_times, bins=30, alpha=0.7)
    plt.xlabel('Wait Time')
    plt.ylabel('Frequency')
    plt.title('Histogram of Wait Times')
    plt.grid()
    plt.savefig(filename)
    plt.close()

def multirun(n, mu, until, policy='JSQd', d=2):
    lambda_n = n - np.sqrt(n) * 2.0  # Arrival rate in halfin whit regime

    # Set up the environment
    uniqueid = f'{n=}_{until=}_{policy=}_{d=}'
    output_dir = os.path.join(os.getcwd(), uniqueid)
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    
    # Run Simulation for JSQ
    env = Environment()
    load_balancer = LoadBalancer(env, n, mu, policy=policy, d=d)
    env.process(arrival_process(env, load_balancer, lambda_n))

    # Progress bar for simulation
    print(f"Running simulation for JSQ with {n} servers, until={until}")
    if until > 99:
        for t in trange(10,until,5):
            env.run(until=t)
    env.run(until=until)

    # Convert queue stats to DataFrame
    queue_stats_df = queue_stats_to_df(load_balancer.queue_stats)
    queue_stats_df.to_csv(f'queue_stats_jsq.csv', index=False)

    # Plot queue stats over time    
    plot_queue_stats_over_time(queue_stats_df, 'JSQ', ylim=n*1.1)

    # print the mean wait time
    wait_times = load_balancer.wait_times
    mean_wait_time = np.mean(wait_times)
    throughput = len(wait_times) / env.now
    L = throughput * mean_wait_time
    print(f'Mean wait time: {mean_wait_time}')
    print(f'Throughput: {throughput}')
    print(f'L = {L}')
    # save the last three print statements to a file
    with open('output.txt', 'w') as f:
        f.write(f'Mean wait time: {mean_wait_time}\n')
        f.write(f'Throughput: {throughput}\n')
        f.write(f'L = {L}\n')

        f.write(f'number of servers: {n}\n')
        f.write(f'service rate: {mu}\n')
        f.write(f'arrival rate: {lambda_n}\n')
        f.write(f'simulation time: {until}\n')
        f.write(f'policy: JSQ\n')

    # Plot mean wait time histogram
    plot_wait_time_histogram(load_balancer.wait_times)


def main():
    # set directory to the Ignore/output folder
    os.chdir(os.path.join(os.getcwd(), 'Ignore', 'output'))
    # Create the output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    home_dir = os.getcwd()

    # Simulation Parameters
    n = 1000   # Number of servers
    mu = 1.0  # Service rate per server
    lambda_n = n - np.sqrt(n) * 2.0  # Arrival rate in halfin whit regime
    until = 1000 # Simulation time

    ds = [2, 4, 8, 16, 32, 64, 128]
    policy = 'JSQd'
    for d in ds:
        print(f"Running simulation for JSQd with {n} servers, until={until}, d={d}")
        multirun(n, mu, until, policy=policy, d=d)
        # change back to home directory
        os.chdir(home_dir)
    # Move all output files to the output directory

    # Make sound notification on macOS only
    if platform.system() == "Darwin":  # Only on macOS
        os.system('afplay /System/Library/Sounds/Glass.aiff')

if __name__ == "__main__":
    main()