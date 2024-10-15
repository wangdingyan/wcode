#!/bin/bash
#cd /nfs/amino-home/zhanglabs/BioLiP2
readlink -f $0
FILE=`readlink -f $0`
rootdir=`dirname $FILE`
cd $rootdir
pwd

#### The following scripts need internet connection ####
echo ./1_pdb.sh
./1_pdb.sh

echo ./2_sifts.pl
./2_sifts.pl

#echo ./script/download_csa.pl
#./script/download_csa.pl     # only need to run annually

echo ./3_bindingdb.pl
./3_bindingdb.pl    # only need to run every few months

#echo ./script/download_moad.pl
##./script/download_moad.pl    # only need to run annually
#
echo ./4_ligand.pl
./4_ligand.pl

echo ./5_curate_smiles.pl
./5_curate_smiles.pl

##### This following script does NOT need internet connection ####
echo ./6_curate_pdb.pl
./6_curate_pdb.pl


#### The following script needs internet connection ####
echo ./7_download_pubmed.pl
./7_download_pubmed.pl
#
##### The following script script does NOT need internet connection ####
#echo ./script/curate_ligand.pl
#./script/curate_ligand.pl
#
#echo ./script/make_weekly.pl
#./script/make_weekly.pl
#
#echo ./script/make_nr.pl
#./script/make_nr.pl
#
#echo ./script/make_rSS.pl
#./script/make_rSS.pl
#
#echo ./script/make_EC.pl
#./script/make_EC.pl
#
#echo ./script/curate_GO.pl
#./script/curate_GO.pl
#
#echo ./script/make_foldseek.pl
#./script/make_foldseek.pl
#
##### The following script needs internet connection ####
#echo ./script/download_rhea.pl
#./script/download_rhea.pl