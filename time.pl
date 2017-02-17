

my $dir = 'transcriptions';
my $dir2 = 'transcriptions3';

    opendir(DIR, $dir) or die $!;

    while (my $file = readdir(DIR)) {

        # Use a regular expression to ignore files beginning with a period
        next if ($file =~ m/^\./);

		open(FILE, "$dir/$file");
		open(OUT, ">$dir2/$file");
		while($line = <FILE>)
		{
		  @list = ();
		  @list = split "\;",$line;
		  print OUT "$list[0],$list[1]\n";
		}
		close(FILE);
		close(OUT);

	print "$file\n";

    }

    closedir(DIR);