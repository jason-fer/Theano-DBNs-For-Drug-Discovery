+-------------------------------- Notes Below: From Jason --------------------------------+
PCBA uses CID: to know actives vs inactives, each file must be paired with a respective 
CSV file. 

PCBA Format: 
1-fingerprint files: CID / fingerprint
2-CSV files: CID active/inactive

DUD-E Format:
CID, fingerprint (active / inactive based on file name)

MUV Format: 
CID, fingerprint (active / inactive based on file name)

Tox21 Format:
{active / inactive}, ID, fingerprint


+-------------------------------- Notes Below: From Spencer --------------------------------+

2015-06-13

   Each of the 4 major data sets are provided in separate folders with fingerprints and 
   original structural representations on which the fingerprints were computed using RDKit.

   All fingerprints are provided as 2048 bit strings as ECFP4 (Morgan-type fingerprints 
   with radius of 2). These files contain the molID (whatever was used in experiment) and the
   bit strings. In some cases, the active/inactive (1/0) classification is provided in the
   first space delimited field on the molecule line in the .fps (fingerprints) file.

   The python scripts used for fingerprint generations are provided in each data set folder.



Data Sets:

   Tox21: https://tripod.nih.gov/tox21/challenge/data.jsp#
          -downloaded the 2D SDF for whole set and each assay individually. The individual
           assay files were better for producing the fingerprints since the 'active' and 
           'inactive' designations could be pulled out of these files (not available in the
           file containing the full set).  Downloaded Jun 8, 2015.
        

   MUV: https://www.tu-braunschweig.de/pharmchem/forschung/baumann/muv

        obtained from supplemental data (Additional data file 5) from paper as directed
        on the official MUV website. Downloaded June 8, 2015.

        The paper providing the dataset with SMILES structure reps for the molecules is 
        included in the folder: 

        Sereina Riniker and Gregory A Landrum, Journal of Cheminformatics 2013, 5:26
        http://www.jcheminf.com/content/5/1/26
	"Open-source platform to benchmark fingerprints for ligand-based virtual screening"

        The original MUV paper is also included: 
        
        Rohrer_Baumann_JCIM2009_MUV_data_sets_for_virtScreen_PCBA_data.pdf


   DUD-E: We already had this data set in hand. Downloaded Feb 6, 2015 from the DUD-E website:

          http://dude.docking.org

          Downloading the "all at once" option.
 	  To cite DUD-E, please reference:    Mysinger MM, Carchia M, Irwin JJ, Shoichet BK 
          J. Med. Chem., 2012, Jul 5. doi 10.1021/jm300687e .         
          
 
   PCBA:  PCBA datasets are dose-response assays performed by NCATS Chemical Genomics Center (NCGC)
          PCBA data were downloaded from PubChem BioAssay using the following search limits:

          TotalSidCount from 10000
          ActiveSidCount from 30
          Chemical
          Confirmatory
          Dose-Response
          Target: Single, NCGC

          These limits correspond to the search query (using the ubChem BioAssay Advanced Search Builder):

          (10000[TotalSidCount] : 1000000000[TotalSidCount]) AND (30[ActiveSidCount] : 1000000000[ActiveSidCount]) AND "small_molecule"[filt] AND "doseresponse"[filt] AND 1[TargetCount] AND "NCGC"[SourceName]

          The first step to acquire these data was to run the query specified in the appendix of the paper.
          The query produces a list that is downloaded "pcassay_result.txt," which includes the AIDs for
          the 128 experiments used in the paper we discussed (list of all AIDs recoverd from query)
  
          PCBA Data files were obtained on Jun 8, 2015 from PubChem. The pubchem CID is the identifier used
          for these molecules in the .fps files.


#############################################################################################

2015-05-22 -- Meeting Regarding mmtNN with Tony (Gitter), Michael Newton, 
              Mike Hoffmann and some CS grad students: 

    Vaidhyanathan Venkiteswaran		vvaidhyanathan@cs.wisc.edu
    Jason Feriante			feriante@cs.wisc.edu
    Huikun Zhang                        huikun@stat.wisc.edu
                        		huikun@cs.wisc.edu

    We discussed using massively multi-tasking Neural Networks (mmtNN) for 
    applications in drug discovery--specificially to build classification model
    to assess compound binding profiles to biomolecular targets.

    They need from us the following:

    FP for all molecules in datasets used in Google paper:  PCBA, Tox21, MUV, and DUD-E
    Bindind data for each of these molecules.


    
