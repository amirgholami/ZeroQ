for MODEL in resnet20_cifar100
do
	echo Testing $MODEL ...
	python uniform_test.py 		\
		--dataset=cifar100 		\
		--model=$MODEL 			\
		--batch_size=64 		\
		--test_batch_size=512
done
