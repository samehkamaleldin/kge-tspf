import os
from tqdm import tqdm

kg_dir = "../data/kg"
ohmnet_data_dp = "../data/ohmnet/"
os.makedirs(kg_dir) if not os.path.isdir(kg_dir) else None

# ===========================================================================================
# process ppi information
ppi_data_dp = os.path.join(ohmnet_data_dp, "bio-tissue-networks")
ppi_kg_filepath = os.path.join(kg_dir, "ppi_facts.txt")
ppi_kg_fd = open(ppi_kg_filepath, "w")

ppi_file_list = os.listdir(ppi_data_dp)
for ppi_filename in tqdm(ppi_file_list, desc="Processing tissue specific protein-peotein interactions"):
    tissue_name = ppi_filename.replace(".edgelist", "")
    filepath = os.path.join(ppi_data_dp, ppi_filename)
    tissue_ppi_list = [l.strip().split() for l in open(filepath).readlines()]
    for g1, g2 in tissue_ppi_list:
        g1_label = "GENE:%s" % g1
        g2_label = "GENE:%s" % g2
        ppi_kg_fd.write("%s\t%s\t%s\n" % (g1_label, tissue_name, g2_label))
ppi_kg_fd.close()
# ===========================================================================================


# ===========================================================================================
# process go annotations information
go_data_dp = os.path.join(ohmnet_data_dp, "bio-tissue-labels")
go_file_list = os.listdir(go_data_dp)
go_kg_filepath = os.path.join(kg_dir, "go_facts.txt")
go_kg_fd = open(go_kg_filepath, "w")
go_file_list = [f for f in go_file_list if ".lab" in f]
for go_filename in tqdm(go_file_list, desc="Processing tissue specific GO annotations"):

    # process file name
    note = go_filename.replace(".lab", "")
    tissue_name, go_id = note.split("_GO:")
    go_label = "GO:%s" % go_id

    # read content
    filepath = os.path.join(go_data_dp, go_filename)
    go_annot_list = [l.strip().split() for l in open(filepath).readlines()[1:]]
    for gene_id, flag in go_annot_list:
        gene_label = "GENE:%s" % gene_id
        go_kg_fd.write("%s\t%s\t%s\t%s\n" % (gene_label, tissue_name, go_label, flag))

go_kg_fd.close()
