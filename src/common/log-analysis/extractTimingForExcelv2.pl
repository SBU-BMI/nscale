#!/usr/bin/perl
#OLD.  do not use.
use strict;
use warnings;


my(@filenames)= <d:/PhD/path/adios-analysis/data/updated_summary/*-syntest*.summary.v2.csv>;
my($dataprefix) = "syntest";

foreach my $filename (@filenames) {
	
	 
	# read the whole file as lines
	open(FH, $filename) or die("ERROR: unable to open $filename!\n");
	my(@lines) = <FH>;
	close(FH);
	
	# open the outputfile 
	my($outfile) = $filename;
	$outfile =~ s/\/data\//\/output\//g;
	$outfile =~ s/\.summary\./.extract./g;
	print "$filename => $outfile\n";
	open(FH2, ">$outfile") or die("ERROR: unable to open output file $outfile\n");
	
	# initialize some variables
	my($type);
	my($procCount);
	my($fileCount);
	my(%output) = ();
	my(%params) = ();
	my(@tokens) = ();
	my($count);
	my($useFileWrite) = 0;
	my($out_line) = "";
	
	# initialize the header
	my($header) = "type,procType,procCount,fileCount,appWallT,sumProcWallT,transport,ioSize,ioGroupSize,bufferSize,dataSize,blocking,compression,ost";
	my($eventTypes) = "Other,Compute,Mem IO,GPU mem IO,Network IO,Network wait,File read,File write,ADIOS init,ADIOS open,ADIOS alloc,ADIOS write,ADIOS close,ADIOS finalize";
	
	my(@colNames) = split(",", $header);
	my(@events) = split(',', $eventTypes);
	my($fname);
	for (my $i = 0; $i < scalar(@events); $i++) {
		$fname = "$events[$i]" . " time(ms)";
		push(@colNames, $fname);
		$fname = "$events[$i]" . " Data(GB)";
		push(@colNames, $fname);
		$fname = "$events[$i]" . " PeakTP(GB/s)";
		push(@colNames, $fname);
		$fname = "$events[$i]" . " MeanTP(GB/s)";
		push(@colNames, $fname);
		$fname = "$events[$i]" . " StdevTP(GB/s)";
		push(@colNames, $fname);
	}
	
	# write the headers
	for (my $i = 0; $i < scalar(@colNames); $i++) {
		$colNames[$i] =~ s/^\s+|\s+$//g;
		print FH2 "$colNames[$i],";
	}
	print FH2 "\n";
	
	my($colName);
	my($nCols) = scalar(@colNames); 
#	print "columns = $nCols\n";
	foreach my $line (@lines) {
		
		# read the line and split
		$line =~ s/^\s+|\s+$//g;  # remove the leading and trailing white spaces
		@tokens = split(",", $line);
		$tokens[0] =~ s/^\s+|\s+$//g;
		
		if ($tokens[0] =~ /^EXPERIMENT/ ) {
						
			%params = ();
			$useFileWrite = 0;
			
			$params{type} = "separate";
			my(@tokens1) = $tokens[1] =~ /.+\/$dataprefix\.[np]([0-9]+)\.f([0-9]+)\.([^\.]+)\.b([0-9]+)\.io([0-9]+)\.is([\-]?[0-9]+).data([0-9]+)\.([bn]+)/g;

			if ($tokens[1] =~ /.+\.n[0-9]+\..+/) {
				$params{procCount} = $tokens1[0] * 12;
			} else {
				$params{procCount} = $tokens1[0];  # kfs has 16 cores per node, but we only use "p" reporting.
			}
			$params{fileCount} = $tokens1[1];

			
			#print "line is $line\n";
			#print "token $_\n" foreach @tokens;
			$params{transport} = $tokens1[2];
			$params{bufferSize} = $tokens1[3];
			$params{ioSize} = $tokens1[4];
			$params{ioGroupSize} = ($tokens1[5] == -1 ? "all" : ($tokens1[5] == $params{procCount} ? "all" : $tokens1[5]));
			$params{dataSize} = $tokens1[6];
			$params{blocking} = $tokens1[7];
		     
			$params{ost} = "n";
			$params{compression} = "n";
			
			$tokens[3] =~ s/^\s+|\s+$//g;
			$params{appWallT} = $tokens[3];
			$tokens[5] =~ s/^\s+|\s+$//g;
			$params{sumProcWallT} = $tokens[5];
			
		} else {
			if ($tokens[0] =~ /field/) {
				# header line. write out.
				# write out the previous line
			
				$count = keys %output;
				#print "$_," for (keys %output); print "\n";
				#print "$_," for @colNames; print "\n";
				#print "$_ = $params{$_}," for (keys %params); print "\n";
				
	#			print "count = $count\n";
				if ($count > 0) {
					my(%newout) = (%params, %output);
					$out_line = "";
					foreach $colName (@colNames) {
						#print "col name = $colName val = ". $newout{$colName} ."\n";
						$out_line .= $newout{$colName} . ", ";
					}
					print FH2 "$out_line\n";				
				}
				%output = ();
				
			} else {
				$output{procType} = $tokens[1];
				$output{procType} =~ s/^\s+|\s+$//g;
				my($event) = $tokens[2];
				$event =~ s/^\s+|\s+$//g;
				my($ename) = "$event" . " time(ms)";
				$output{$ename} = $tokens[6];
				$output{$ename} =~ s/^\s+|\s+$//g;
				$ename = "$event" . " Data(GB)";
				$output{$ename} = $tokens[21];
				$output{$ename} =~ s/^\s+|\s+$//g;
				$ename = "$event" . " PeakTP(GB/s)";
				$output{$ename} = $tokens[22];
				$output{$ename} =~ s/^\s+|\s+$//g;
				$ename = "$event" . " MeanTP(GB/s)";
				$output{$ename} = $tokens[23];
				$output{$ename} =~ s/^\s+|\s+$//g;
				$ename = "$event" . " StdevTP(GB/s)";
				$output{$ename} = $tokens[24];
				$output{$ename} =~ s/^\s+|\s+$//g;
				
			}
			
		}
	}
	# header line. write out.
	# write out the previous line

	$count = keys %output;
	#print "count = $count\n";
	if ($count > 0) {
		my(%newout) = (%params, %output);
		$out_line = "";
		foreach $colName (@colNames) {
			#print "col name = $colName val = ". $output{$colName} ."\n";
			$out_line .= $newout{$colName} . ", ";
		}
		print FH2 "$out_line\n";				
	}
	%output = ();
		

	close(FH2);

}
