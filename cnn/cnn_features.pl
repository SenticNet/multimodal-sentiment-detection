for($k=0;$k<10;$k++){
system("perl format1.pl run2/layer0_vid_train$k\.csv > run2/train_cnn$k");
system("perl format1.pl run2/layer0_vid_val$k\.csv > run2/val_cnn$k");
system("perl format1.pl run2/layer0_vid_test$k\.csv > run2/test_cnn$k");
}