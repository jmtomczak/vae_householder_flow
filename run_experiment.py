import run_VAE_HF

print('\n___...----------------- START EXPERIMENT -----------------...___\n')

dataset = ['mnist']
repeats = [0]

for d in dataset:
    print( '----------------- DATASET: %s -----------------\n' % d )

    for r in repeats:
        run_VAE_HF.run('warmup', 200, d, r, mini_batch_size=100, max_epochs=3,
                         number_of_Householders=0, encoder=[300, 300], decoder=[300, 300], number_z=40)

    for r in repeats:
        run_VAE_HF.run('warmup', 200, d, r, mini_batch_size=100, max_epochs=3,
                         number_of_Householders=1, encoder=[300, 300], decoder=[300, 300], number_z=40)

    for r in repeats:
        run_VAE_HF.run('freebits', 0.25, d, r, mini_batch_size=100, max_epochs=3,
                         number_of_Householders=2, encoder=[300, 300], decoder=[300, 300], number_z=40)