$newline = ();
open(FILE,$ARGV[0]);
while($line = <FILE>)
{
 chomp($line);
 @list = split
 $line =~ tr/\[\]//d;
 @list = split " ",$line;
 for($i=0;$i<scalar(@list);$i++){$newline.=$list[$i].",";}
}
close(FILE);
print $newline;