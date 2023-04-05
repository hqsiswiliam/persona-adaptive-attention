from sacrebleu import BLEU

paa_file_path = "generated_text/PAA-small.txt"

encoder_gpt_file_path = "generated_text/encoder_gpt.txt"

gpt_small_file_path = "generated_text/gpt2-small.txt"

gpt_medium_file_path = "generated_text/gpt2-medium.txt"

gpt_large_file_path = "generated_text/gpt-large.txt"


def get_sacre_bleu(file_path):
    with open(file_path, 'r') as file:
        results = file.readlines()

    targets = [line[4:].strip() for line in results if line.startswith('GT:')]
    preds = [line[4:].strip() for line in results if line.startswith('PD:')]
    bleu = BLEU().corpus_score(preds, [targets])
    return bleu


encoder_gpt_bleu = get_sacre_bleu(encoder_gpt_file_path)
gpt_small_bleu = get_sacre_bleu(gpt_small_file_path)
gpt_medium_bleu = get_sacre_bleu(gpt_medium_file_path)
gpt_large_bleu = get_sacre_bleu(gpt_large_file_path)
paa_bleu = get_sacre_bleu(paa_file_path)

#
#
print("{:.2f} & {:.2f}".format(encoder_gpt_bleu.precisions[0],encoder_gpt_bleu.precisions[1]))
print("{:.2f} & {:.2f}".format(gpt_small_bleu.precisions[0],gpt_small_bleu.precisions[1]))
print("{:.2f} & {:.2f}".format(gpt_medium_bleu.precisions[0],gpt_medium_bleu.precisions[1]))
print("{:.2f} & {:.2f}".format(gpt_large_bleu.precisions[0],gpt_large_bleu.precisions[1]))
print("{:.2f} & {:.2f}".format(paa_bleu.precisions[0],paa_bleu.precisions[1]))
