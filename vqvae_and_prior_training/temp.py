with open('vqvae_gpt_prior_training.py', 'r') as f_in, open('output.txt', 'w') as f_out:
    for line in f_in:
        if len(line.rstrip('\n')) <= 500:
            f_out.write(line)
