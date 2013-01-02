#!/usr/bin/perl

use strict;
use warnings;


my(@filenames)= <../data/keeneland-syntest-params2/*.csv>;
my($dataprefix) = "syntest";


foreach my $filename (@filenames) {
	
	 
	# read the whole file as lines
	open(FH, $filename) or die("ERROR: unable to open $filename!\n");
	my(@lines) = <FH>;
	close(FH);
	
	my($outfile) = $filename;
	$outfile =~ s/data\//output\//g;
	$outfile =~ s/summary/extract/g;
	print "filename = $outfile\n";
	open(FH2, ">$outfile") or die("ERROR: unable to open output file $outfile\n");
	
	my($type);
	my($procCount);
	my($fileCount);
	my(%output) = ();
	my(@tokens) = ();
	my($count);
	my($useFileWrite) = 0;
	my($out_line) = "";
	my($header) = "type,procCount,fileCount,startT,endT,transport,ioSize,ioGroupSize,bufferSize,dataSize,blocking,compression,ost,computePercent,readPercent,readTP,writePercent,writeTP";
	my(@colNames) = split(",", $header);
	
	for (my $i = 0; $i < scalar(@colNames); $i++) {
		$colNames[$i] =~ s/^\s+|\s+$//g;
	}
	
	my($colName);
	my($nCols) = scalar(@colNames); 
#	print "columns = $nCols\n";
	print FH2 "$header\n";
	foreach my $line (@lines) {
		
		$line =~ s/^\s+|\s+$//g;  # remove the leading and trailing white spaces
		
		if ($line =~ /^\/home\/tcpan\/PhD\/path\/Data\/adios\// ) {
			# write out the previous line
			
			$count = keys %output;
#			print "count = $count\n";
			if ($count > 0) {
				$out_line = "";
				foreach $colName (@colNames) {
#					print "col name = $colName val = ". $output{$colName} ."\n";
					$out_line .= $output{$colName} . ", ";
				}
				print FH2 "$out_line\n";				
			}
						
			%output = ();
			$useFileWrite = 0;
			
			@tokens = $line =~ /.+\/$dataprefix\.[np]([0-9]+)\.f([0-9]+).*/g;
			#print "token $_\n" foreach @tokens;
			$output{type} = "separate";
			if ($line =~ /.+\.n[0-9]+\..+/) {
				$output{procCount} = $tokens[0] * 12;
			} else {
				$output{procCount} = $tokens[0];
			}
			$output{fileCount} = $tokens[1];
			
				@tokens = $line =~ /.+\/$dataprefix\..+\.f[0-9]+\.([^\.]+)\.b([0-9]+)\.io([0-9]+)\.is([\-]?[0-9]+).data([0-9]+)\.([bn]+)/g;
				#print "line is $line\n";
				#print "token $_\n" foreach @tokens;
				$output{transport} = $tokens[0];
				$output{bufferSize} = $tokens[1];
				$output{ioSize} = $tokens[2];
				$output{ioGroupSize} = ($tokens[3] == -1 ? "all" : ($tokens[3] == $output{procCount} ? "all" : $tokens[3]));
				$output{dataSize} = $tokens[4];
				$output{blocking} = $tokens[5];
			
				$output{ost} = "n";
				$output{compression} = "n";
			
		} elsif ($line =~ /^start time,/) {
			@tokens = split(",", $line);
			$tokens[1] =~ s/^\s+|\s+$//g;
			$output{startT} = $tokens[1];
			$tokens[3] =~ s/^\s+|\s+$//g;
			$output{endT} = $tokens[3];
		} elsif ($line =~ /^File read,/) {
			@tokens = split(",", $line);
			$tokens[2] =~ s/^\s+|\s+$//g;
			$output{readPercent} = $tokens[2];
			$tokens[9] =~ s/^\s+|\s+$//g;
			$output{readTP} = $tokens[9];
		} elsif ($line =~ /^File write,/) {
			@tokens = split(",", $line);
			if ($useFileWrite == 0) {
				$tokens[2] =~ s/^\s+|\s+$//g;
				$output{writePercent} = $tokens[2];
				$tokens[9] =~ s/^\s+|\s+$//g;
				$output{writeTP} = $tokens[9];
				$useFileWrite = 1;
			} else {
				print "File write: output writePercent and WriteTP set to $output{writePercent} and $output{writeTP} previously set by $useFileWrite.  ignore new values $tokens[2], $tokens[9]\n";
			}
		} elsif ($line =~ /^save image,/) {
			@tokens = split(",", $line);
			if ($useFileWrite == 0) {
				$tokens[2] =~ s/^\s+|\s+$//g;
				$output{writePercent} = $tokens[2];
				$tokens[9] =~ s/^\s+|\s+$//g;
				$output{writeTP} = $tokens[9];
				$useFileWrite = 2;
			} else {
				print "save image: output writePercent and WriteTP set to $output{writePercent} and $output{writeTP} previously set by $useFileWrite.  ignore new values $tokens[2], $tokens[9]\n";
			}
		} elsif ($line =~ /^IO POSIX Write,/) {
			@tokens = split(",", $line);
			if ($useFileWrite == 0) {
				$tokens[2] =~ s/^\s+|\s+$//g;
				$output{writePercent} = $tokens[2];
				$tokens[9] =~ s/^\s+|\s+$//g;
				$output{writeTP} = $tokens[9];
				$useFileWrite = 3;
			} else {
				print "IO POSIX Write: output writePercent and WriteTP set to $output{writePercent} and $output{writeTP} previously set by $useFileWrite.  ignore new values $tokens[2], $tokens[9]\n";
			}
		} elsif ($line =~ /^Adios,/) {
			@tokens = split(",", $line);
			if ($useFileWrite == 0) {
				$tokens[2] =~ s/^\s+|\s+$//g;
				$output{writePercent} = $tokens[2];
				$tokens[9] =~ s/^\s+|\s+$//g;
				$output{writeTP} = $tokens[9];
				$useFileWrite = 4;
				
			} else {
				print "Adios: output writePercent and WriteTP set to $output{writePercent} and $output{writeTP} previously set by $useFileWrite.  ignore new values $tokens[2], $tokens[9]\n";
			}
		} elsif ($line =~ /^Compute,/) {
			@tokens = split(",", $line);
			$tokens[2] =~ s/^\s+|\s+$//g;
			$output{computePercent} = $tokens[2];
		}
		
	}
		
	# write out the previous line
	$count = keys %output;
	#print "count = $count\n";
	if ($count > 0) {
		$out_line = "";
		foreach $colName (@colNames) {
			$out_line .= $output{$colName} . ", ";
		}
		print FH2 "$out_line\n";				
	}

	close(FH2);

}
