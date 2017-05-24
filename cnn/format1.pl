open(FILE,$ARGV[0]);
while($line = <FILE>)
{
 chomp($line);
 @list = ();
 @list = split(",",$line);

 for($i=0;$i<scalar(@list);$i++)
 {
    $list[$i] =~ s/["\[\]]//g;
    @list2 = ();
    @list2 = split " ",$list[$i];
    for($i2=0;$i2<scalar(@list2);$i2++)
    { print $list2[$i2]."\n";}
 } 

}
close(FILE);


