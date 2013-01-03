#!/usr/bin/perl

use strict;
use warnings;


sub appendData ($$$$$) {
	use strict;
	use warnings;

	my($outp, $headers, $nodeType, $eventType, $tokens) = @_;
	my($cn);
	$cn = $nodeType . ' ' . $eventType . ' min count';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[10];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	$cn = $nodeType . ' ' . $eventType . ' min time(ms)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[11];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	$cn = $nodeType . ' ' . $eventType . ' min data(MB)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[12];
	$outp->{$cn} =~ s/^\s+|\s+$//g;

	$cn = $nodeType . ' ' . $eventType . ' max count';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[13];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	$cn = $nodeType . ' ' . $eventType . ' max time(ms)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[14];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	$cn = $nodeType . ' ' . $eventType . ' max data(MB)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[15];
	$outp->{$cn} =~ s/^\s+|\s+$//g;

	$cn = $nodeType . ' ' . $eventType . ' mean count';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[16];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	$cn = $nodeType . ' ' . $eventType . ' mean time(ms)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[17];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	$cn = $nodeType . ' ' . $eventType . ' mean data(MB)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[18];
	$outp->{$cn} =~ s/^\s+|\s+$//g;

	$cn = $nodeType . ' ' . $eventType . ' stdev count';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[19];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	$cn = $nodeType . ' ' . $eventType . ' stdev time(ms)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[20];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	$cn = $nodeType . ' ' . $eventType . ' stdec data(MB)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[21];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	
}



my(@filenames)= </home/tcpan/PhD/path/Data/adios/*-syntest*.summary.v2.csv>;
my($dataprefix) = "syntest";




foreach my $filename (@filenames) {
	
	 
	# read the whole file as lines
	open(FH, $filename) or die("ERROR: unable to open $filename!\n");
	my(@lines) = <FH>;
	close(FH);
	
	
	# initialize some variables
	my($type);
	my($nProcs);
	my($fileCount);
	my(%output) = ();
	my(%params) = ();
	my(@tokens) = ();
	my($count);
	my($out_line) = "";
	
	# initialize the header
	my($header) = "type,nProcs,nMaster,nCompute,nIO,fileCount,transport,ioGroupSize,bufferSize,dataSize,blocking,compression,ost,appWallT,sumProcWallT";
	# node types are (m/assign), io, seg, w(=io,seg)
	my(@colNames) = split(",", $header);




	my(@outputs);
	my(%colNames2);

	# get the data	
	foreach my $line (@lines) {
		
		# read the line and split
		$line =~ s/^\s+|\s+$//g;  # remove the leading and trailing white spaces
		@tokens = split(",", $line);
		$tokens[0] =~ s/^\s+|\s+$//g;
		
		if ($tokens[0] =~ /^EXPERIMENT/ ) {
						
			# first write out.
			# write out the previous line
		
			$count = keys %output;
			#print "$_," for (keys %output); print "\n";
			#print "$_," for @colNames; print "\n";
			#print "$_ = $params{$_}," for (keys %params); print "\n";
			
#			print "count = $count\n";
			if ($count > 0) {
				my(%newout) = (%params, %output);
				push(@outputs, \%newout);
			}

			# then reset with new parameter, etc.
			%output = ();
			%params = ();
			
			$params{type} = "syn sep";
			my(@tokens1) = $tokens[1] =~ /.+\/$dataprefix\.[np]([0-9]+)\.f([0-9]+)\.([^\.]+)\.b([0-9]+)\.io([0-9]+)\.is([\-]?[0-9]+).data([0-9]+)\.([bn]+)/g;

			if ($tokens[1] =~ /.+\.n[0-9]+\..+/) {
				$params{nProcs} = $tokens1[0] * 12;
			} else {
				$params{nProcs} = $tokens1[0];  # kfs has 16 cores per node, but we only use "p" reporting.
			}
			$params{fileCount} = $tokens1[1];

			
			#print "line is $line\n";
			#print "token $_\n" foreach @tokens;
			$params{transport} = $tokens1[2];
			$params{bufferSize} = $tokens1[3];
			$params{nIO} = $tokens1[4];
			$params{ioGroupSize} = ($tokens1[5] == -1 ? $params{nIO} : $tokens1[5]);
			$params{dataSize} = $tokens1[6];
			$params{blocking} = $tokens1[7];
		     
			$params{ost} = "n";
			$params{compression} = "n";
			
			$tokens[3] =~ s/^\s+|\s+$//g;
			$params{appWallT} = $tokens[3];
			$tokens[5] =~ s/^\s+|\s+$//g;
			$params{sumProcWallT} = $tokens[5];
			
		} else {
			$tokens[1] =~ s/^\s+|\s+$//g;
			$tokens[2] =~ s/^\s+|\s+$//g;
			
			
			if ($tokens[1] =~ /m|assign/) {
				if ($tokens[2] =~ /File read/) {
					appendData(\%output, \%colNames2, $tokens[1], $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS init/) {
					$params{nMaster} = $tokens[3];
				}
			} elsif ($tokens[1] =~ /w|seg/) {

				if ($tokens[2] =~ /File read|Compute/) {
					appendData(\%output, \%colNames2, $tokens[1], $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS init/) {
					$params{nCompute} = $tokens[3];
				}

			} elsif ($tokens[1] =~ /w|io/) {

				if ($tokens[2] =~ /File write/) {
					appendData(\%output, \%colNames2, $tokens[1], $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS open/) {
					appendData(\%output, \%colNames2, $tokens[1], $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS alloc/) {
					appendData(\%output, \%colNames2, $tokens[1], $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS write/) {
					appendData(\%output, \%colNames2, $tokens[1], $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS close/) {
					appendData(\%output, \%colNames2, $tokens[1], $tokens[2], \@tokens);
				}

			} else {
				# all, or header.  ignore.			
				next;
			}

			if ($tokens[2] =~ /Mem IO/) {
				appendData(\%output, \%colNames2, $tokens[1], $tokens[2], \@tokens);
			} elsif ($tokens[2] =~ /Network IO/) {
				appendData(\%output, \%colNames2, $tokens[1], $tokens[2], \@tokens);
			} elsif ($tokens[2] =~ /Network wait/) {
				appendData(\%output, \%colNames2, $tokens[1], $tokens[2], \@tokens);
			} elsif ($tokens[2] =~ /Network IO NB/) {
				appendData(\%output, \%colNames2, $tokens[1], $tokens[2], \@tokens);
			} elsif ($tokens[2] =~ /ADIOS init/) {
				appendData(\%output, \%colNames2, $tokens[1], $tokens[2], \@tokens);
			} elsif ($tokens[2] =~ /ADIOS finalize/) {
				appendData(\%output, \%colNames2, $tokens[1], $tokens[2], \@tokens);
			} else {
				next;
			}
		}
	}
	# header line. write out.
	# write out the previous line

	$count = keys %output;
	#print "$_," for (keys %output); print "\n";
	#print "$_," for @colNames; print "\n";
	#print "$_ = $params{$_}," for (keys %params); print "\n";

	#			print "count = $count\n";
	if ($count > 0) {
		my(%newout) = (%params, %output);
		push(@outputs, \%newout);
	}
	

	# open the outputfile 
	my($outfile) = $filename;
	$outfile =~ s/\/adios\//\/adios-analysis\//g;
	$outfile =~ s/\.summary\./.extract./g;
	print "$filename => $outfile\n";
	open(FH2, ">$outfile") or die("ERROR: unable to open output file $outfile\n");
	
	foreach my $colName (@colNames) {
		print FH2 $colName . ", ";
	}
	foreach my $colName (sort (keys %colNames2)) {
		print FH2 $colName . ", ";
	}
	print FH2 "\n";
	my($op);
	foreach $op (@outputs) {
		$out_line = "";
		foreach my $colName (@colNames) {
			if (exists($op->{$colName})) {
				$out_line .= $op->{$colName};
			}
			$out_line .= ", ";
		}

		foreach my $colName (sort (keys %colNames2)) {
			#print "col name = $colName val = ". $newout{$colName} ."\n";
			if (exists($op->{$colName})) {
				$out_line .= $op->{$colName};
			}
			$out_line .= ", ";
		}
		print FH2 "$out_line\n";				
	}

		
	close(FH2);



}
