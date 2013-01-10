#!/usr/bin/perl
package cci_common_filenameparser;
use strict;
use warnings;

BEGIN {
	require Exporter;

	our @ISA = qw( Exporter );
	our @EXPORT = qw( parseFileName );
	our @EXPORT_OK = qw( parseFileName );
}

sub parseFileName($$$) {
	use strict;
	use warnings;

	my ( $filetoken, $appTime, $cpuTime ) = @_;

	# initialize some variables
	my (%params) = ();
	my (@tokens1);

	$filetoken =~ s/^\s+|\s+$//g;
	$appTime   =~ s/^\s+|\s+$//g;
	$cpuTime   =~ s/^\s+|\s+$//g;

	$params{appWallT} = $appTime / 1000;    # wall time was reported in microsec
	$params{sumProcWallT} =
	  $cpuTime / 1000;                      # wall time was reported in microsec

	# then reset with new parameter, etc.
	$params{filename}    = $filetoken;
	$params{type}        = "TCGA";
	$params{layout}      = "separate";
	$params{sys}         = "keeneland";
	$params{nMaster}     = 1;
	$params{nCompute}    = "all";
	$params{nIO}         = "all";
	$params{transport}   = "opencv";
	$params{bufferSize}  = 1;
	$params{ioGroupSize} = "all";
	$params{dataSize}    = 4096;
	$params{blocking}    = "b";
	$params{compression} = "n";
	$params{ost}         = "n";

	if ( $filetoken =~ /kfs/ ) {
		$params{sys} = "kfs";
	}
	elsif ( $filetoken =~ /jaguar/ ) {
		$params{sys} = "jaguar";
	}

	if ( $filetoken =~ /push/ ) {
		$params{nMaster} = 0;
	}

	if ( $filetoken =~ m/\.ost/ ) {
		$params{ost} = "y";
	}

	if ( $filetoken =~ m/syntest/ ) {


		$params{type} = "synth";

		if ( $filetoken =~
/.+\/[^\/\.]+\.[np][0-9]+\.f[\-]?[0-9]+\.[^\.]+\.b[0-9]+\.io[\-]?[0-9]+\.is[\-]?[0-9]+\.data[0-9]+\.[bn]+$/
		  )
		{
			@tokens1 = $filetoken =~
/.+\/([^\/\.]+)\.[np]([0-9]+)\.f([\-]?[0-9]+)\.([^\.]+)\.b([0-9]+)\.io([\-]?[0-9]+)\.is([\-]?[0-9]+)\.data([0-9]+)\.([bn]+)$/;


			if ( $filetoken =~ /.+\.n[0-9]+\..+/ ) {
				$params{nProcs} = $tokens1[1] * 12;
			}
			else {
				$params{nProcs} = $tokens1[1]
				  ;  # kfs has 16 cores per node, but we only use "p" reporting.
			}
			$params{fileCount} = $tokens1[2];

			$params{transport}  = $tokens1[3];
			$params{bufferSize} = $tokens1[4];
			$params{nIO}        = $tokens1[5];
			$params{nCompute} =
			  $params{nProcs} - $params{nMaster} - $params{nIO};
			$params{ioGroupSize} =
			  ( $tokens1[6] == -1
				? "all"
				: ( $tokens1[6] == $tokens1[5] ? "all" : $tokens1[6] ) );
			$params{dataSize} = $tokens1[7];
			$params{blocking} = $tokens1[8];

		}
		else {
			print "not matched in syntest: $filetoken\n";
			return undef;
		}
	}
	elsif ( $filetoken =~ m/tcga.p2048.kfs/ ) {

		# blocking

		if ( $filetoken =~
/.+\/tcga-[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+\.[^\.]+\.b[0-9]+\.io[\-]?[0-9]+\.is[\-]?[0-9]+\.data[0-9]+\.[nb]+[-]?.*$/
		  )
		{

# /home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1/TCGA.separate.keeneland.n32.f9600.MPI_AMR.b4.io60-1.is60[.ost]
			@tokens1 = $filetoken =~
/.+\/tcga-([^\.]+)\.[np]([0-9]+)\.f([\-]?[0-9]+)\.([^\.]+)\.b([0-9]+)\.io([\-]?[0-9]+)\.is([\-]?[0-9]+)\.data([0-9]+)\.([nb]+)[-]?.*$/;

			$params{layout} =
			  ( $tokens1[0] =~ /sep/ ? "separate" : $tokens1[0] );
			if ( $filetoken =~ /.+\.n[0-9]+\..+/ ) {
				$params{nProcs} = $tokens1[1] * 12;
			}
			else {
				$params{nProcs} = $tokens1[1];
			}
			$params{fileCount} = $tokens1[2];

			$params{transport}  = $tokens1[3];
			$params{bufferSize} = $tokens1[4];
			$params{nIO}        = ( $tokens1[5] == -1 ? "all" : $tokens1[5] );
			$params{nCompute} =
			  $params{nProcs} -
			  $params{nMaster} -
			  ( $params{nIO} =~ /all/ ? 0 : $params{nIO} );
			$params{ioGroupSize} =
			  ( $tokens1[6] == -1
				? "all"
				: ( $tokens1[6] == $tokens1[5] ? "all" : $tokens1[6] ) );
			$params{dataSize} = $tokens1[7];
			$params{blocking} = $tokens1[8];

		}
		else {
			print "not matched in tcga.kfs: $filetoken\n";
			return undef;
		}

	}
	elsif ( $filetoken =~ m/-hpdc2012-/ ) {
		if ( $filetoken =~ m/compressed|-compress-/ ) {
			$params{compression} = "y";
		}

		# blocking

		if ( $filetoken =~
/.+\/TCGA\.separate\.[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+\.[^\.]+\.b[0-9]+\.io[\-]?[0-9]+-1\.is[\-]?[0-9]+[\.]?[ost]*$/
		  )
		{

# /home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1/TCGA.separate.keeneland.n32.f9600.MPI_AMR.b4.io60-1.is60[.ost]
			@tokens1 = $filetoken =~
/.+\/TCGA\.(separate)\.[^\.]+\.[np]([0-9]+)\.f([\-]?[0-9]+)\.([^\.]+)\.b([0-9]+)\.io([\-]?[0-9]+)-1\.is([\-]?[0-9]+)[\.]?[ost]*$/;

			$params{transport}  = $tokens1[3];
			$params{bufferSize} = $tokens1[4];
			$params{nIO}        = $tokens1[5];
			$params{ioGroupSize} =
			  ( $tokens1[6] == -1
				? "all"
				: ( $tokens1[6] == $tokens1[5] ? "all" : $tokens1[6] ) );

		}
		elsif ( $filetoken =~
/.+\/TCGA\.coloc\.[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+\.[^\.]+\.b[0-9]+\.is[\-]?[0-9]+[\.]?[ost]*.*$/
		  )
		{
			@tokens1 = $filetoken =~
/.+\/TCGA\.(coloc)\.[^\.]+\.[np]([0-9]+)\.f([\-]?[0-9]+)\.([^\.]+)\.b([0-9]+)\.is([\-]?[0-9]+)[\.]?[ost]*-.*$/;

# /home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1/TCGA.coloc.keeneland.n60.f18000.MPI_LUSTRE.b4.is-1[.ost]-MPI_LUSTRE
#									print ".2\n";

			$params{transport}  = $tokens1[3];
			$params{bufferSize} = $tokens1[4];
			$params{ioGroupSize} =
			  ( $tokens1[5] == -1
				? "all"
				: ( $tokens1[5] == $tokens1[4] ? "all" : $tokens1[5] ) );

		}
		elsif ( $filetoken =~
			/.+\/TCGA\.baseline\.[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+[\.]?[ost]*$/ )
		{
			@tokens1 = $filetoken =~
/.+\/TCGA\.(baseline)\.[^\.]+\.[np]([0-9]+)\.f([\-]?[0-9]+)[\.]?[ost]*$/;

# /home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1/TCGA.baseline.keeneland.n32.f9600[.ost]
#									print ".3\n";

		}
		else {
			print "not matched in hpdc2012: $filetoken\n";
			return undef;
		}
		$params{layout} = $tokens1[0];
		if ( $filetoken =~ /.+\.n[0-9]+\..+/ ) {
			$params{nProcs} = $tokens1[1] * 12;
		}
		else {
			$params{nProcs} = $tokens1[1];
		}
		$params{fileCount} = $tokens1[2];
		$params{nCompute} =
		  $params{nProcs} -
		  $params{nMaster} -
		  ( $params{nIO} =~ /all/ ? 0 : $params{nIO} );

	}
	elsif ( $filetoken =~ m/-nb-vs-b/ ) {

		# blocking

		if ( $filetoken =~ /\.nb/ ) {
			$params{blocking} = "nb";
		}

		if ( $filetoken =~
/.+\/TCGA\.separate\.[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+\.[^\.]+\.b[0-9]+\.io[\-]?[0-9]+-1\.is[\-]?[0-9]+[\.]?[nb]*$/
		  )
		{

# /home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1/TCGA.separate.keeneland.n32.f9600.MPI_AMR.b4.io60-1.is60[.nb]
			@tokens1 = $filetoken =~
/.+\/TCGA\.(separate)\.[^\.]+\.[np]([0-9]+)\.f([\-]?[0-9]+)\.([^\.]+)\.b([0-9]+)\.io([\-]?[0-9]+)-1\.is([\-]?[0-9]+)[\.]?[nb]*$/;

			$params{transport}  = $tokens1[3];
			$params{bufferSize} = $tokens1[4];
			$params{nIO}        = $tokens1[5];
			$params{ioGroupSize} =
			  ( $tokens1[6] == -1
				? "all"
				: ( $tokens1[6] == $tokens1[5] ? "all" : $tokens1[6] ) );

		}
		elsif ( $filetoken =~
/.+\/TCGA\.coloc\.[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+\.[^\.]+\.b[0-9]+\.is[\-]?[0-9]+[\.]?[nb]*-.*$/
		  )
		{

# /home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1/TCGA.coloc.keeneland.n60.f18000.MPI_LUSTRE.b4.is-1[.nb]-MPI_LUSTRE
#									print ".2\n";
			@tokens1 = $filetoken =~
/.+\/TCGA\.(coloc)\.[^\.]+\.[np]([0-9]+)\.f([\-]?[0-9]+)\.([^\.]+)\.b([0-9]+)\.is([\-]?[0-9]+)[\.]?[nb]*-.*$/;
			$params{transport}  = $tokens1[3];
			$params{bufferSize} = $tokens1[4];
			$params{ioGroupSize} =
			  ( $tokens1[5] == -1
				? "all"
				: ( $tokens1[5] == $tokens1[4] ? "all" : $tokens1[5] ) );

		}
		elsif ( $filetoken =~
			/.+\/TCGA\.baseline\.[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+[\.]?[nb]*$/ )
		{

# /home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1/TCGA.baseline.keeneland.n32.f9600.nb
#									print ".3\n";
			@tokens1 = $filetoken =~
/.+\/TCGA\.(baseline)\.[^\.]+\.[np]([0-9]+)\.f([\-]?[0-9]+)[\.]?[nb]*$/;
		}
		else {
			print "not matched in hpdc2012: $filetoken\n";
			return undef;
		}
		$params{layout} = $tokens1[0];
		if ( $filetoken =~ /.+\.n[0-9]+\..+/ ) {
			$params{nProcs} = $tokens1[1] * 12;
		}
		else {
			$params{nProcs} = $tokens1[1];
		}
		$params{fileCount} = $tokens1[2];
		$params{nCompute} =
		  $params{nProcs} -
		  $params{nMaster} -
		  ( $params{nIO} =~ /all/ ? 0 : $params{nIO} );

	}
	elsif ( $filetoken =~ m/jaguar-tcga-transports/ ) {
		if ( $filetoken =~ m/sep_osts\// ) {
			$params{ost} = "y";
		}

		if ( $filetoken =~
/.+\/TCGA\.separate\.[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+\.[^\.]+\.b[0-9]+\.io[\-]?[0-9]+-[0-9]+\.is[\-]?[0-9]+-1$/
		  )
		{

# /home/tcpan/PhD/path/Data/adios/jaguar-tcga-transports-sep_osts/TCGA.separate.jaguar.p10240.f100000.MPI_AMR.b4.io2560-16.is16-1
#									print ".1\n";
			@tokens1 = $filetoken =~
/.+\/TCGA\.(separate)\.[^\.]+\.[np]([0-9]+)\.f([\-]?[0-9]+)\.([^\.]+)\.b([0-9]+)\.io([\-]?[0-9]+)-[0-9]+\.is([\-]?[0-9]+)-1$/;
			$params{transport}  = $tokens1[3];
			$params{bufferSize} = $tokens1[4];
			$params{nIO}        = $tokens1[5];
			$params{ioGroupSize} =
			  ( $tokens1[6] == -1
				? "all"
				: ( $tokens1[6] == $tokens1[5] ? "all" : $tokens1[6] ) );

		}
		elsif ( $filetoken =~
			/.+\/TCGA\.baseline\.[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+$/ )
		{

# /home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1/TCGA.baseline.keeneland.n32.f9600
#									print ".3\n";
			@tokens1 = $filetoken =~
			  /.+\/TCGA\.(baseline)\.[^\.]+\.[np]([0-9]+)\.f([\-]?[0-9]+)$/;
		}
		else {
			print "not matched in hpdc2012: $filetoken\n";
			return undef;
		}
		$params{layout} = $tokens1[0];
		if ( $filetoken =~ /.+\.n[0-9]+\..+/ ) {
			$params{nProcs} = $tokens1[1] * 12;
		}
		else {
			$params{nProcs} = $tokens1[1];
		}
		$params{fileCount} = $tokens1[2];
		$params{nCompute} =
		  $params{nProcs} -
		  $params{nMaster} -
		  ( $params{nIO} =~ /all/ ? 0 : $params{nIO} );

	}
	elsif ( $filetoken =~ m/-strong/ ) {

		if ( $filetoken =~
/.+\/TCGA\.separate\.[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+\.[^\.]+\.b[0-9]+\.io[\-]?[0-9]+-[0-9]+\.is[\-]?[0-9]+-1$/
		  )
		{

# /home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1/TCGA.separate.keeneland.n32.f9600.MPI_AMR.b4.io60-1.is60-1
#									print ".1\n";
			@tokens1 = $filetoken =~
/.+\/TCGA\.(separate)\.[^\.]+\.[np]([0-9]+)\.f([\-]?[0-9]+)\.([^\.]+)\.b([0-9]+)\.io([\-]?[0-9]+)-[0-9]+\.is([\-]?[0-9]+)-1$/;

			$params{layout}     = "separate";
			$params{transport}  = $tokens1[3];
			$params{bufferSize} = $tokens1[4];
			$params{nIO}        = $tokens1[5];
			$params{ioGroupSize} =
			  ( $tokens1[6] == -1
				? "all"
				: ( $tokens1[6] == $tokens1[5] ? "all" : $tokens1[6] ) );

		}
		elsif ( $filetoken =~
/.+\/TCGA\.co-loc\.[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+\.[^\.]+\.is[\-]?[0-9]+-1-.*$/
		  )
		{

# /home/tcpan/PhD/path/Data/adios/jaguar-tcga-strong1/TCGA.co-loc.jaguar.p10240.f100000.MPI_AMR.is1-1-MPI_AMR
#									print ".2\n";
			@tokens1 = $filetoken =~
/.+\/TCGA\.(co-loc)\.[^\.]+\.[np]([0-9]+)\.f([\-]?[0-9]+)\.([^\.]+)\.is([\-]?[0-9]+)-1-.*$/;

			$params{layout}      = "coloc";
			$params{transport}   = $tokens1[3];
			$params{ioGroupSize} = ( $tokens1[4] == -1 ? "all" : $tokens1[4] );
		}
		elsif ( $filetoken =~
			/.+\/TCGA\.baseline\.[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+$/ )
		{

# /home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1/TCGA.baseline.keeneland.n32.f9600
#									print ".3\n";
			@tokens1 = $filetoken =~
			  /.+\/TCGA\.(baseline)\.[^\.]+\.[np]([0-9]+)\.f([\-]?[0-9]+)$/;

			$params{layout} = "baseline";
		}
		else {
			print "not matched in hpdc2012: $filetoken\n";
			return undef;
		}

		if ( $filetoken =~ /.+\.n[0-9]+\..+/ ) {
			$params{nProcs} = $tokens1[1] * 12;
		}
		else {
			$params{nProcs} = $tokens1[1];
		}
		$params{fileCount} = $tokens1[2];
		$params{nCompute} =
		  $params{nProcs} -
		  $params{nMaster} -
		  ( $params{nIO} =~ /all/ ? 0 : $params{nIO} );

	}
	elsif ( $filetoken =~ m/randtime/) {
		# /home/tcpan/PhD/path/Data/adios/PreIPROBE/keeneland-compressed-randtime/TCGA.separate.keeneland.n60.f18000.MPI_AMR.b4.io60-1.is60
				if ( $filetoken =~
/.+\/TCGA\.separate\.[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+\.[^\.]+\.b[0-9]+\.io[\-]?[0-9]+-[0-9]+\.is[\-]?[0-9]+$/
		  )
		{

			@tokens1 = $filetoken =~
/.+\/TCGA\.(separate)\.[^\.]+\.[np]([0-9]+)\.f([\-]?[0-9]+)\.([^\.]+)\.b([0-9]+)\.io([\-]?[0-9]+)-[0-9]+\.is([\-]?[0-9]+)$/;

			$params{layout}     = "separate";
			$params{transport}  = $tokens1[3];
			$params{bufferSize} = $tokens1[4];
			$params{nIO}        = $tokens1[5];
			$params{ioGroupSize} =
			  ( $tokens1[6] == -1
				? "all"
				: ( $tokens1[6] == $tokens1[5] ? "all" : $tokens1[6] ) );

		}
		elsif ( $filetoken =~
/.+\/TCGA\.co-loc\.[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+\.[^\.]+\.is[\-]?[0-9]+-.*$/
		  )
		{

# /home/tcpan/PhD/path/Data/adios/jaguar-tcga-strong1/TCGA.co-loc.jaguar.p10240.f100000.MPI_AMR.is1-1-MPI_AMR
#									print ".2\n";
			@tokens1 = $filetoken =~
/.+\/TCGA\.(co-loc)\.[^\.]+\.[np]([0-9]+)\.f([\-]?[0-9]+)\.([^\.]+)\.is([\-]?[0-9]+)-1-.*$/;

			$params{layout}      = "coloc";
			$params{transport}   = $tokens1[3];
			$params{ioGroupSize} = ( $tokens1[4] == -1 ? "all" : $tokens1[4] );
		}
		elsif ( $filetoken =~
			/.+\/TCGA\.baseline\.[^\.]+\.[np][0-9]+\.f[\-]?[0-9]+$/ )
		{

# /home/tcpan/PhD/path/Data/adios/keeneland-hpdc2012-compressed1/TCGA.baseline.keeneland.n32.f9600
#									print ".3\n";
			@tokens1 = $filetoken =~
			  /.+\/TCGA\.(baseline)\.[^\.]+\.[np]([0-9]+)\.f([\-]?[0-9]+)$/;

			$params{layout} = "baseline";
		}
		else {
			print "not matched in hpdc2012: $filetoken\n";
			return undef;
		}

		if ( $filetoken =~ /.+\.n[0-9]+\..+/ ) {
			$params{nProcs} = $tokens1[1] * 12;
		}
		else {
			$params{nProcs} = $tokens1[1];
		}
		$params{fileCount} = $tokens1[2];
		$params{nCompute} =
		  $params{nProcs} -
		  $params{nMaster} -
		  ( $params{nIO} =~ /all/ ? 0 : $params{nIO} );

		
	}
	else {
		print "not matched: $filetoken\n";
		return undef;
	}

	return \%params;
}

#
### testing code.
#use strict;
#use warnings;
#
#
#my(@filenames)= </home/tcpan/PhD/path/Data/adios/*.summary.v2.csv>;
## initialize the header
#my($header) = "type,layout,sys,nProcs,nMaster,nCompute,nIO,fileCount,transport,ioGroupSize,bufferSize,dataSize,blocking,compression,ost,appWallT,sumProcWallT";
## node types are (m/assign), io, seg, w(=io,seg)
#my(@colNames) = split(",", $header);
#
#my($outfile) = "./testparse.csv";
#open(FH2, ">$outfile") or die("ERROR: unable to open output file $outfile\n");
#
#print FH2 "filename,";
#foreach my $colName (@colNames) {
#	print FH2 $colName . ",";
#}
#print FH2 "\n";
#
#
#foreach my $filename (@filenames) {
#
#
#	# read the whole file as lines
#	open(FH, $filename) or die("ERROR: unable to open $filename!\n");
#	my(@lines) = <FH>;
#	close(FH);
#
#	foreach my $line (@lines) {
#
#		# read the line and split
#		$line =~ s/^\s+|\s+$//g;  # remove the leading and trailing white spaces
#		my(@tokens) = split(",", $line);
#		$tokens[0] =~ s/^\s+|\s+$//g;
#		
#		if ($tokens[0] =~ /^EXPERIMENT/ ) {
#			my($parameters) = parseFileName($tokens[1], $tokens[3], $tokens[5]);
#			
#			if (defined($parameters)) {
#				
#				print FH2 "$tokens[1],";
#				print FH2 "$parameters->{$_}," foreach (@colNames);
#				print FH2 "\n";
#			}
#		}
#	}
#	
#}
#close(FH2);

1;	