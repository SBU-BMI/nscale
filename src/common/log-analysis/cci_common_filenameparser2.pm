#!/usr/bin/perl
package cci_common_filenameparser2;
use strict;
use warnings;

BEGIN {
	require Exporter;

	our @ISA = qw( Exporter );
	our @EXPORT = qw( parseFileName );
	our @EXPORT_OK = qw( parseFileName );
}

sub parseFileName($) {
	use strict;
	use warnings;

	my ( $filetoken ) = @_;

	# initialize some variables
	my (%params) = ();

	$filetoken =~ s/^\s+|\s+$//g;

	# then reset with new parameter, etc.
	$params{filename}    = $filetoken;
	$params{type}        = $filetoken =~ /syntest|synthetic/ ? "synth" : "TCGA-GBM" ;
	$params{layout}      = $filetoken =~ /baseline/ ? "baseline" : ($filetoken =~ /coloc|co-loc/ ? "coloc" : ($filetoken =~ /-sep\.r/ ? "sep-RW" : "sep-W"));
	print "$filetoken\n" if $filetoken !~ /\.[np]\d+\./;
	$params{nProcs}		 = $filetoken =~ /\.n(\d+)\./ ? $1 * 12 : ($filetoken =~ /\.p(\d+)\./ ? $1 : undef);
	$params{nMaster}     = $filetoken =~ /push|r(\d+)\./ ? 0 : 1;
	$params{nWrite}      = $filetoken =~ /\.io(\d+)(-\d+)?\./ ? $1 : ($params{nProcs} - $params{nMaster});  # io only in separate.  io-1 is same as using all nodes (only for coloc and baseline)
	$params{nRead}       = $filetoken =~ /\.r(\d+)\./ ? $1 : $filetoken =~ /\.r\./ ? 16 : $params{nProcs} - $params{nMaster} - ($params{layout} =~ /sep-/ ? $params{nWrite} : 0); 
		# r only in separate and with io present
	$params{nCompute}    = $params{nProcs} - $params{nMaster} - ($params{layout} =~ /sep-/ ? $params{nWrite} : 0) - ($params{layout} =~ /sep-RW/ ? $params{nRead} : 0);
	$params{transport}   = $filetoken =~ /\.f[\-]?\d+\.([^\.]+)\.b\d+\./ ? $1 : "opencv";
	$params{bufferSize}  = $filetoken =~ /\.b(\d+)\./ ? $1 : 1;
	$params{ioGroupSize} = $filetoken =~ /\.is(\d+)(-\d+)?/ ? $1 : $params{nWrite};  # is-1 is same as not having isXX.
	$params{dataSize}    = $filetoken =~ /\.data(\d+)\./ ? $1 : 4096;
	$params{blocking}    = $filetoken =~ /\.nb$|\.nb-/ ? 0 : 1;
	$params{compression} = $filetoken =~ /compressed|-compress-/ ? 1 : 0;
	$params{ost}         = $filetoken =~ /\.ost$|\.ost-|sep-osts/ ? 1 : 0;
	$params{fileCount}	 = $filetoken =~ /\.f([\-]?\d+)/ ? $1 : undef;
	$params{MPIIprobe}	 = $filetoken =~ /PreIPROBE/ ? 1 : 0;

	if ( $filetoken =~ /kfs/ ) {
		$params{sys} = "kfs";
	} elsif ( $filetoken =~ /kids/ ) {
		$params{sys} = "kids";
	} elsif ( $filetoken =~ /jaguar/ ) {
		$params{sys} = "jaguar";
	} elsif ( $filetoken =~ /titan/ ) {
		$params{sys} = "titan";
	} elsif ( $filetoken =~ /keeneland/ ) {
		$params{sys} = "kids";
	} else {
		$params{sys} = "unknown";
	}

	if ( $filetoken =~ /PreIPROBE\/[^\/]+-syntest\.run\d/ ) {

		#/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-syntest.run2/synthetic.datasizes.p720.push.NULL.2048
		#/home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-syntest.run2/synthetic.datasizes.p720.NULL.2048
		if ( $filetoken =~ /.+\/synthetic\.datasizes\.p\d+\.(push\.)?([^\.]+)\.(\d+)$/ ) {
			$params{transport}  = $1;
			$params{dataSize} = $2;

		} else {
			print STDERR "not matched in syntest: $filetoken\n";
			return undef;
		}
	} elsif ( $filetoken =~ /syntest/ ) {
		#/home/tcpan/PhD/path/Data/adios/kfs-syntest-param1/syntest.p2048.f4096.POSIX.b4.io512.is1.data512[.b|.nb]

		# all separate
		if ( $filetoken !~ /.+\/syntest\.[np]\d+\.f[\-]?\d+\.[^\.]+\.b\d+\.io[\-]?\d+\.is[\-]?\d+\.data\d+\.[nb]+$/ ) {
			print STDERR "not matched in syntest: $filetoken\n";
			return undef;
		}
	} elsif ( $filetoken =~ /tcga\.p2048\.kfs|tcga\.titan\.p\d+/ ) {
		# /home/tcpan/PhD/path/Data/adios/tcga.p2048.kfs.1/tcga-sep.p2048.f10240.MPI_LUSTRE.b4.io1536.is15.data4096.nb
		# /home/tcpan/PhD/path/Data/adios/tcga.p2048.kfs.1/tcga-coloc.p2048.f10240.MPI_AMR.b4.io-1.is15.data4096.b-MPI_AMR
		# /home/tcpan/PhD/path/Data/adios/tcga.p2048.kfs.1/tcga-baseline.p2048.f10240.opencv.b0.io-1.is-1.data4096.b
		# /home/tcpan/PhD/path/Data/adios/tcga.titan.p2048.1/tcga-sep.r.p2048.f10240.MPI_AMR.b4.io1536.is15.data4096.nb
		# /home/tcpan/PhD/path/Data/adios/tcga.titan.p10240.1/tcga-sep.r300.p2048.f10240.MPI_AMR.b4.io1536.is15.data4096.nb
		if ( ($filetoken !~ /.+\/tcga-[^\.]+\.[np]\d+\.f[\-]?\d+\.[^\.]+\.b\d+\.io[\-]?\d+\.is[\-]?\d+\.data\d+\.[nb]+(-.*)?$/ ) && 
			($filetoken !~ /.+\/tcga-[^\.]+\.r\d*\.[np]\d+\.f[\-]?\d+\.[^\.]+\.b\d+\.io[\-]?\d+\.is[\-]?\d+\.data\d+\.[nb]+(-.*)?$/ ) ) {
			print STDERR "not matched in tcga.kfs: $filetoken\n";
			return undef;
		}

	} elsif ( $filetoken =~ /-hpdc2012-|-nb-vs-b/ ) {

		# /home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1/TCGA.separate.keeneland.n32.f9600.MPI_AMR.b4.io60-1.is60[.ost]
		# /home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1/TCGA.coloc.keeneland.n60.f18000.MPI_LUSTRE.b4.is-1[.ost]-MPI_LUSTRE
		# /home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1/TCGA.baseline.keeneland.n32.f9600[.ost]
		# /home/tcpan/PhD/path/Data/adios/keeneland-nb-vs-b/TCGA.separate.keeneland.n60.f18000.POSIX.b4.io60-1.is15[.nb]
		# /home/tcpan/PhD/path/Data/adios/keeneland-nb-vs-b/TCGA.coloc.keeneland.n60.f18000.MPI_LUSTRE.b4.is-1[.nb]-MPI_LUSTRE
		# /home/tcpan/PhD/path/Data/adios/keeneland-nb-vs-b/TCGA.baseline.keeneland.n32.f9600[.nb]
		if ( ($filetoken !~ /.+\/TCGA\.separate\.[^\.]+\.[np]\d+\.f[\-]?\d+\.[^\.]+\.b\d+\.io[\-]?\d+-1\.is[\-]?\d+(\.ost|\.osts)?(\.nb|\.b)?$/ ) &&
			( $filetoken !~ /.+\/TCGA\.coloc\.[^\.]+\.[np]\d+\.f[\-]?\d+\.[^\.]+\.b\d+\.is[\-]?\d+(\.ost|\.osts)?(\.nb|\.b)?(-.*)?$/ ) &&
			( $filetoken !~ /.+\/TCGA\.baseline\.[^\.]+\.[np]\d+\.f[\-]?\d+(\.ost|\.osts)?(\.nb|\.b)?$/ ) ) {

			print STDERR "not matched in hpdc2012 or nb-vs-b: $filetoken\n";
			return undef;
		}

	} elsif ( $filetoken =~ m/jaguar-tcga-transports|-strong/ ) {

		# /home/tcpan/PhD/path/Data/adios/jaguar-tcga-transports-sep_osts/TCGA.separate.jaguar.p10240.f100000.MPI_AMR.b4.io2560-16.is16-1
		# /home/tcpan/PhD/path/Data/adios/jaguar-tcga-transports-sep_osts/TCGA.baseline.jaguar.p10240.f100000
		#/home/tcpan/PhD/path/Data/adios/PreIPROBE/jaguar-tcga-strong1/TCGA.separate.jaguar.p10240.f100000.MPI_AMR.b4.io7680-16.is16-1
		#/home/tcpan/PhD/path/Data/adios/PreIPROBE/jaguar-tcga-strong1/TCGA.co-loc.jaguar.p10240.f100000.MPI_AMR.is1-1-MPI_AMR
		#/home/tcpan/PhD/path/Data/adios/PreIPROBE/jaguar-tcga-strong1/TCGA.baseline.jaguar.p5120.f100000

		if ( ( $filetoken !~ /.+\/TCGA\.separate\.[^\.]+\.[np]\d+\.f[\-]?\d+\.[^\.]+\.b\d+\.io[\-]?\d+-\d+\.is[\-]?\d+-1$/ ) &&
			( $filetoken !~ /.+\/TCGA\.co-loc\.[^\.]+\.[np]\d+\.f[\-]?\d+\.[^\.]+\.is[\-]?\d+-1(-.*)?$/ ) &&
			( $filetoken !~ /.+\/TCGA\.baseline\.[^\.]+\.[np]\d+\.f[\-]?\d+$/ ) ) {

			print STDERR "not matched in jaguar-transport or -string: $filetoken\n";
			return undef;
		}

	} elsif ( $filetoken =~ m/randtime/) {
		# /home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-compressed-randtime/TCGA.separate.keeneland.n60.f18000.MPI_AMR.b4.io60-1.is60
		if ( $filetoken !~ /.+\/TCGA\.separate\.[^\.]+\.[np]\d+\.f[\-]?\d+\.[^\.]+\.b\d+\.io[\-]?\d+-\d+\.is[\-]?\d+$/ ) {
			print STDERR "not matched in randtime: $filetoken\n";
			return undef;
		}
		
	} else {
		print STDERR "not matched: $filetoken\n";
		return undef;
	}

	return \%params;
}

#
### testing code.
# use strict;
# use warnings;


# # initialize the header
# my($needHeader) = 1;

# my($outfile) = "./testparse2.csv";
# open(FH2, ">$outfile") or die("ERROR: unable to open output file $outfile\n");

# my(@filenames)= </home/tcpan/PhD/path/Data/adios/*-*/*.csv>;
# push(@filenames, </home/tcpan/PhD/path/Data/adios/*.*/*.csv>);
# push(@filenames, </home/tcpan/PhD/path/Data/adios/PreIPROBE/*/*.csv>);

# foreach my $filename (@filenames) {

	# $filename =~ s/\.csv$//;
	# my($parameters) = parseFileName($filename);

	# if (!defined($parameters)) {
		# print "ERROR with parsing $filename\n";
		# next;
	# }
	
	# my($first) = 1;
	# if ( $needHeader == 1 ) {
		# foreach my $colName (sort keys %{$parameters}) {
			# print FH2 "," if ($first == 0);
			# print FH2 $colName;
			# $first = 0;
		# }
		# print FH2 "\n";
	
		# $needHeader = 0;
	# }

	# $first = 1;
	# foreach my $colName (sort keys %{$parameters}) {
		# print FH2 "," if ($first == 0);
		# print FH2 $parameters->{$colName} if (defined($parameters->{$colName}));  # if undefined, leave blank.
		# print "$filename $colName undefined\n" if (!defined($parameters->{$colName}));
		# $first = 0;
	# }
	# print FH2 "\n";
	
# }

# close(FH2);

1;
