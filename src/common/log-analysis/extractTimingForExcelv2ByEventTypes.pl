#!/usr/bin/perl

use strict;
use warnings;
use lib ('.');
use cci_common_filenameparser;

my(@filenames)= </home/tcpan/PhD/path/Data/adios/PreIPROBE/*.summary.v2.csv>;


sub appendData ($$$$$) {
	use strict;
	use warnings;

	my($outp, $headers, $nodeType, $eventType, $tokens) = @_;
	my($cn);
	$cn = $nodeType . ' ' . $eventType . ' count';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[3];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	$cn = $nodeType . ' ' . $eventType . ' min time(ms)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[4];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	$cn = $nodeType . ' ' . $eventType . ' max time(ms)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[5];
	$outp->{$cn} =~ s/^\s+|\s+$//g;

	$cn = $nodeType . ' ' . $eventType . ' total time(ms)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[6];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	$cn = $nodeType . ' ' . $eventType . ' mean time(ms)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[7];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	$cn = $nodeType . ' ' . $eventType . ' stdev(ms)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[8];
	$outp->{$cn} =~ s/^\s+|\s+$//g;

	$cn = $nodeType . ' ' . $eventType . ' total data(MB)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[9];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	$cn = $nodeType . ' ' . $eventType . ' peakTP(GB/s)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[22];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	$cn = $nodeType . ' ' . $eventType . ' meanTP(GB/s)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[23];
	$outp->{$cn} =~ s/^\s+|\s+$//g;

	$cn = $nodeType . ' ' . $eventType . ' stdevTP(GB/s)';
	$headers->{$cn} = 1;
	$outp->{$cn} = $tokens->[24];
	$outp->{$cn} =~ s/^\s+|\s+$//g;
	
}

sub fixComputeDataSize ($$$$) {
	use strict;
	use warnings;

	my($outp, $chunkSize, $nodeType, $eventType) = @_;
	if ($eventType =~ /Compute/) {
		my($cn);
		$cn = $nodeType . ' ' . $eventType . ' total data(MB)';
		$outp->{$cn} *= $chunkSize;
	
		$cn = $nodeType . ' ' . $eventType . ' peakTP(GB/s)';
		$outp->{$cn} *= $chunkSize;
	
		$cn = $nodeType . ' ' . $eventType . ' meanTP(GB/s)';
		$outp->{$cn} *= $chunkSize;
	
		$cn = $nodeType . ' ' . $eventType . ' stdevTP(GB/s)';
		$outp->{$cn} *= $chunkSize;
	}
}




# initialize the header
my($header) = "type,layout,sys,nProcs,nMaster,nCompute,nIO,fileCount,transport,ioGroupSize,bufferSize,dataSize,blocking,compression,ost,appWallT,sumProcWallT";
# node types are (m/assign), io, seg, w(=io,seg)
my(@colNames) = split(",", $header);


foreach my $filename (@filenames) {
	
	 
	# read the whole file as lines
	open(FH, $filename) or die("ERROR: unable to open $filename!\n");
	my(@lines) = <FH>;
	close(FH);
	
	
	# initialize some variables
	my(%output) = ();
	my($params);
	my(@tokens) = ();
	my($count);
	my($out_line) = "";
	
	my(@outputs);
	my(%colNames2);
	my($ignore) = 0;

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
			
#			print "count = $count\n";
			if ($count > 0) {
				my(%newout) = (%{$params}, %output);
				push(@outputs, \%newout);
			}

			# then reset with new parameter, etc.
			%output = ();
			$params = parseFileName($tokens[1], $tokens[3], $tokens[5]);
			if (defined($params)) {
				$ignore = 0;
			} else {
				$ignore = 1;
				print "IGNORING $tokens[1]\n";
			}
			
		} else {
			if ($ignore == 1) {
				next;
			}
			
			$tokens[1] =~ s/^\s+|\s+$//g;
			$tokens[2] =~ s/^\s+|\s+$//g;
			
			if ($tokens[1] =~ /m|assign/) {
				if ($tokens[2] =~ /File read/) {
					appendData(\%output, \%colNames2, "assign", $tokens[2], \@tokens);
					
				} elsif ($tokens[2] =~ /Mem IO/) {
					appendData(\%output, \%colNames2, "assign", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /Network IO/) {
					appendData(\%output, \%colNames2, "assign", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /Network wait/) {
					appendData(\%output, \%colNames2, "assign", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /Network IO NB/) {
					appendData(\%output, \%colNames2, "assign", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS init/) {
					appendData(\%output, \%colNames2, "assign", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS finalize/) {
					appendData(\%output, \%colNames2, "assign", $tokens[2], \@tokens);
				} else {
					next;
				}
				
			} elsif ($tokens[1] =~ /w/) {

				

				if ($tokens[2] =~ /File read/) {
					appendData(\%output, \%colNames2, "seg", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /Compute/) {
					appendData(\%output, \%colNames2, "seg", $tokens[2], \@tokens);
					fixComputeDataSize(\%output, $params->{dataSize} * $params->{dataSize} * 4, "seg", $tokens[2]);
					
				} elsif ($tokens[2] =~ /File write/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS open/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS alloc/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS write/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS close/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
					
				} elsif ($tokens[2] =~ /Mem IO/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /Network IO/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /Network wait/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /Network IO NB/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS init/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS finalize/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} else {
					next;
				}
				

				# memory io, network, and adios times for "w" are in the IO part of the reporting.

			} elsif ($tokens[1] =~ /seg/) {
				
				if ($tokens[2] =~ /File read/) {
					appendData(\%output, \%colNames2, "seg", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /Compute/) {
					appendData(\%output, \%colNames2, "seg", $tokens[2], \@tokens);
					fixComputeDataSize(\%output, $params->{dataSize} * $params->{dataSize} * 4, "seg", $tokens[2]);
					
				} elsif ($tokens[2] =~ /Mem IO/) {
					appendData(\%output, \%colNames2, "seg", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /Network IO/) {
					appendData(\%output, \%colNames2, "seg", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /Network wait/) {
					appendData(\%output, \%colNames2, "seg", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /Network IO NB/) {
					appendData(\%output, \%colNames2, "seg", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS init/) {
					appendData(\%output, \%colNames2, "seg", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS finalize/) {
					appendData(\%output, \%colNames2, "seg", $tokens[2], \@tokens);
				} else {
					next;
				}

			} elsif ($tokens[1] =~ /io/) {

				if ($tokens[2] =~ /File write/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS open/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS alloc/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS write/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS close/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
					
				} elsif ($tokens[2] =~ /Mem IO/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /Network IO/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /Network wait/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /Network IO NB/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS init/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} elsif ($tokens[2] =~ /ADIOS finalize/) {
					appendData(\%output, \%colNames2, "io", $tokens[2], \@tokens);
				} else {
					next;
				}

				

			} else {
				# all, or header.  ignore.			
				next;
			}

		}
	}
	# header line. write out.
	# write out the previous line

	$count = keys %output;

	#			print "count = $count\n";
	if ($count > 0) {
		my(%newout) = (%{$params}, %output);
		push(@outputs, \%newout);
	}
	

	# open the outputfile 
	my($outfile) = $filename;
	$outfile =~ s/\/adios\//\/adios-analysis\//g;
	$outfile =~ s/\.summary\./\.extract\.events\./g;
	print "$filename => $outfile\n";
	open(FH2, ">$outfile") or die("ERROR: unable to open output file $outfile\n");
	
	print FH2 "$_," foreach (@colNames);
	print FH2 "$_," foreach (sort (keys %colNames2));
	print FH2 "\n";
	
	my($op);
	foreach $op (@outputs) {
		$out_line = "";
		foreach my $colName (@colNames) {
			if (exists($op->{$colName})) {
				$out_line .= $op->{$colName};
			}
			$out_line .= ",";
		}

		foreach my $colName (sort (keys %colNames2)) {
			if (exists($op->{$colName})) {
				$out_line .= $op->{$colName};
			}
			$out_line .= ",";
		}
		print FH2 "$out_line\n";				
	}

		
	close(FH2);



}
