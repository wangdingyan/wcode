#!/usr/bin/perl
# download binding affinity from BindingDB and PDBbind-CN
use strict;
use File::Basename;
use Cwd 'abs_path';
my $rootdir = "/mnt/d/WDrugDataset";

print "download BindingDB\n";
system("mkdir -p $rootdir/bind");
system("wget -q --no-check-certificate 'https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp' -O $rootdir/bind/Download.jsp");
if (`cat $rootdir/bind/Download.jsp`=~/(BindingDB_All_\w+.tsv.zip)/)
{
    my $outfile="$1";
    # system("wget -q --no-check-certificate 'https://www.bindingdb.org/rwd/bind/downloads/$outfile' -O $rootdir/bind/BindingDB_All_tsv.zip");
    if (-s "$rootdir/bind/BindingDB_All_tsv.zip")
    {
        # system("zcat $rootdir/bind/BindingDB_All_tsv.zip| cut -f2,9-14,28,29,43,44,48|  head -1 > $rootdir/bind/BindingDB_All_Activity.tsv");
        system("zcat $rootdir/bind/BindingDB_All_tsv.zip| cut -f2,9-14,28,29,43,44,48|  grep -P '^\\S+' | grep -vP '\\t\\t\\t\$'  >> $rootdir/bind/BindingDB_All_Activity.tsv");
        # &gzipFile("$rootdir/bind/BindingDB_All_Activity.tsv");
    }
}

#  0 Ligand SMILES
#  1 Ki (nM)
#  2 IC50 (nM)
#  3 Kd (nM)
#  4 EC50 (nM)
#  5 kon (M-1-s-1)
#  6 koff (s-1)
#  7 Ligand HET ID in PDB
#  8 PDB ID(s) for Ligand-Target Complex
#  9 UniProt (SwissProt) Primary ID of Target Chain
#  10 UniProt (SwissProt) Secondary ID(s) of Target Chain
#  11 UniProt (TrEMBL) Primary ID of Target Chain


sub gzipFile
{
    my ($filename)=@_;
    my $oldNum=`zcat $filename.gz 2>/dev/null|wc -l`+0;
    my $newNum=` cat $filename   |wc -l`+0;
    if (0.8*$oldNum>$newNum)
    {
        print "WARNING! do not update $filename from $oldNum to $newNum entries\n";
        return;
    }
    print "update $filename from $oldNum to $newNum entries\n";
    system("gzip -f $filename");
    return;
}

