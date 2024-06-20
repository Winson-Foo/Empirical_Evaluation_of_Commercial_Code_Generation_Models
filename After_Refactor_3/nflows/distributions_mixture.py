# main.py
from distributions import MADEMoG

def main():
    # Example usage
    mog = MADEMoG(features=10, hidden_features=5, context_features=2)
    samples = mog.sample(num_samples=5, context=None)
    log_probs = mog.log_prob(inputs=samples, context=None)
    print(samples)
    print(log_probs)


if __name__ == "__main__":
    main()