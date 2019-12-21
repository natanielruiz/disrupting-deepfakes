import glob

output_txt = 'list_attr_mydataset.txt'

for idx, f in enumerate(glob.glob('./my-processed-dataset/*.csv')):
    with open(f, 'r') as csv_file:
        csv_file.readline()
        csv_list = csv_file.readline().split(', ')
        if float(csv_list[1]) >= 0.88:
            aus = " ".join(csv_list[2:19])
            open(output_txt, 'a').write(f.split('/')[-1].split('.')[0] + '.jpg ' + aus + '\n')