#!/bin/bash


#https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/kernel_custom.html

DEVDIR=$(pwd)

export KERNEL_SRC=$DEVDIR/Linux_for_Tegra/source/public/kernel/kernel-4.9
export BUILD_KERNEL=$DEVDIR/build_linux-4.9
export CROSS_COMPILE=$DEVDIR/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-
export ARCH=arm64
export DTC_FLAGS='--symbols'

#UPLOAD_SSH=jnano@192.168.88.224
UPLOAD_SSH=jnano@192.168.1.179

UTS_RELEASE=`cat $BUILD_KERNEL/include/generated/utsrelease.h | sed -e 's/.*"\(.*\)".*/\1/' `
DIR_SAVE_MODULE="modules_$UTS_RELEASE"

#SSH_ORANGE=root@192.168.1.105
#SSHFS_DIR=$DEVDIR/sshfs_dir
#ROOT_DIR=/media/tymbys/f7c6ac5c-af9b-431d-873d-95e3f81ea1841
#BOOT_DIR=/media/tymbys/8dee508f-0b7c-41e0-99a9-2489393b767e

#sudo umount $SSHFS_DIR

#DTC=/usr/bin/dtc

[ ! -d "$SSHFS_DIR" ] && mkdir -p "$SSHFS_DIR"



pushd $KERNEL_SRC

KERNEL_VERSION=$(make kernelversion)

echo "KERNEL_VERSION: $KERNEL_VERSION"
echo "-----------------------"


read -p "make tegra_defconfig? y/n : " -rsn1 val
echo "$val"
if [ "$val" = "y" ]; then
	echo "Start make tegra_defconfig"
	make  O=${BUILD_KERNEL} tegra_defconfig
fi

read -p "make menuconfig? y/n : " -rsn1 val
echo "$val"
if [ "$val" = "y" ]; then
	echo "Start make menuconfig"
	make  O=${BUILD_KERNEL} menuconfig
fi

make -j4 O=${BUILD_KERNEL}
#make -j4 O=${BUILD_KERNEL} zImage 
#make -j4 O=${BUILD_KERNEL} modules
#make -j4 O=${BUILD_KERNEL} dtbs


echo "KERNEL_VERSION: $KERNEL_VERSION"
echo "-----------------------"

read -p "Upload to temp rootfs? y/n : " -rsn1 val
echo "$val"
if [ "$val" = "y" ]; then
	echo "Start Upload"

	sudo rm -rf $DEVDIR/tmp_RF

	#sudo chown tymbys:tymbys $ROOT_DIR
	#scp $BUILD_KERNEL/vmlinux $SSH_ORANGE:~

	# sudo sshfs -o allow_other $SSH_ORANGE:/ $SSHFS_DIR
	#sudo cp $BUILD_KERNEL/arch/arm64/boot/Image $BOOT_DIR/Image_$KERNEL_VERSION
	#sudo make O=${BUILD_KERNEL} INSTALL_MOD_PATH=$ROOT_DIR 

	[ ! -d "$DEVDIR/tmp_RF" ] && mkdir -p "$DEVDIR/tmp_RF"
	[ ! -d "$DEVDIR/tmp_RF/boot" ] && mkdir -p "$DEVDIR/tmp_RF/boot"

	cp $BUILD_KERNEL/arch/arm64/boot/Image $DEVDIR/tmp_RF/boot
	cp $BUILD_KERNEL/arch/arm64/boot/dts/freescale/fsl-imx8mq-phanbell.dtb $DEVDIR/tmp_RF/boot

	# mkdir $DEVDIR/tmp_RF
	make O=${BUILD_KERNEL} INSTALL_MOD_PATH=$DEVDIR/tmp_RF modules_install
	#echo sudo cp -r $DEVDIR/tmp_RF/* $ROOT_DIR/
	#sudo cp -r $DEVDIR/tmp_RF/* $ROOT_DIR/

	#[ ! -d "$SSHFS_DIR/boot/dtb-$KERNEL_VERSION" ] && mkdir -p "$SSHFS_DIR/boot/dtb-$KERNEL_VERSION"
	#[ ! -d "$SSHFS_DIR/boot/dtb-$KERNEL_VERSION/overlay" ] && mkdir -p "$SSHFS_DIR/boot/dtb-$KERNEL_VERSION/overlay"

	#sudo cp -r $BUILD_KERNEL/arch/arm64/boot/dts/freescale/fsl-imx8mq-phanbell.dtb $BOOT_DIR/

	
	#sudo umount $SSHFS_DIR

	# pushd $SSHFS_DIR/boot/

	# sudo unlink dtb
	# sudo ln -s dtb-$KERNEL_VERSION dtb

	# sudo unlink zImage
	# sudo ln -s zImage_$KERNEL_VERSION zImage

	# popd

	# sudo umount $SSHFS_DIR

	#scp -pr $SSHFS_DIR/lib/ $SSH_ORANGE:/
	read -p "Upload modules to $UPLOAD_SSH ? y/n : " -rsn1 val
	echo "$val"
	if [ "$val" = "y" ]; then
		echo "Start Upload"

		#echo scp -r $DEVDIR/tmp_RF/lib $UPLOAD_SSH:~/$DIR_SAVE_MODULE
		rsync -Wav --progress $DEVDIR/tmp_RF/lib $UPLOAD_SSH:~/$DIR_SAVE_MODULE
		ssh $UPLOAD_SSH "sudo cp -rf ~/$DIR_SAVE_MODULE/lib /"

	fi

fi




read -p "Upload Image to $UPLOAD_SSH ? y/n : " -rsn1 val
echo "$val"
if [ "$val" = "y" ]; then

	UTS_RELEASE=`cat $BUILD_KERNEL/include/generated/utsrelease.h | sed -e 's/.*"\(.*\)".*/\1/' `
	DIR_SAVE_MODULE="modules_$UTS_RELEASE"

	ssh $UPLOAD_SSH "[ ! -d  \"./$DIR_SAVE_MODULE/\" ] && mkdir -p \"./$DIR_SAVE_MODULE/\""
	scp $BUILD_KERNEL/arch/arm64/boot/Image $UPLOAD_SSH:~/$DIR_SAVE_MODULE
	ssh $UPLOAD_SSH "sudo cp ~/$DIR_SAVE_MODULE/Image /boot/Image_$UTS_RELEASE"
	
fi


read -p "Upload DTB to $UPLOAD_SSH ? y/n : " -rsn1 val
echo "$val"
if [ "$val" = "y" ]; then

	# UTS_RELEASE=`cat $BUILD_KERNEL/include/generated/utsrelease.h | sed -e 's/.*"\(.*\)".*/\1/' `
	# DIR_SAVE_MODULE="modules_$UTS_RELEASE"
	FILE_DTB=tegra210-p3448-0000-p3449-0000-b00.dtb
	EXPORT_DTB=$BUILD_KERNEL/arch/arm64/boot/dts/$FILE_DTB

	ssh $UPLOAD_SSH "[ ! -d  \"./$DIR_SAVE_MODULE/\" ] && mkdir -p \"./$DIR_SAVE_MODULE/\""
	scp $EXPORT_DTB $UPLOAD_SSH:~/$DIR_SAVE_MODULE
	ssh $UPLOAD_SSH "sudo cp ~/$DIR_SAVE_MODULE/$FILE_DTB /boot/"

fi

popd
