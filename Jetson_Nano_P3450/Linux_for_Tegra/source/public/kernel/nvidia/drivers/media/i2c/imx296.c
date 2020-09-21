/*
 * imx296.c - imx296 sensor driver for nvidia jetson
 */

#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/gpio.h>
#include <linux/module.h>
#include <linux/seq_file.h>
#include <linux/of.h>
#include <linux/of_device.h>
#include <linux/of_gpio.h>

#include <media/tegra_v4l2_camera.h>
#include <media/tegracam_core.h>

#include <media/camera_common.h>

#define IMX296_DIGITAL_GAIN_MIN		0
#define IMX296_DIGITAL_GAIN_MAX		480
#define IMX296_DIGITAL_GAIN_DEFAULT	200

#define IMX296_DIGITAL_EXPOSURE_MIN	29
#define IMX296_DIGITAL_EXPOSURE_MAX	15110711
#define IMX296_DIGITAL_EXPOSURE_DEFAULT	10000

#define IMX296_TABLE_WAIT_MS	0
#define IMX296_TABLE_END	1

#define imx296_reg struct reg_8

static int sensor_mode;
module_param(sensor_mode, int, 0644);
MODULE_PARM_DESC(sensor_mode, "Sensor Mode: 0=10bit_stream 1=10bit_ext_trig");

static imx296_reg imx296_start_stream[] = {
	{0x3000, 0x00},		/* mode select streaming on */
	{0x300A, 0x00},		/* mode select streaming on */
	{IMX296_TABLE_END, 0x00}
};

static imx296_reg imx296_stop_stream[] = {

	{0x300A, 0x01},		/* mode select streaming off */
	{0x3000, 0x01},		/* mode select streaming off */
	{IMX296_TABLE_END, 0x00}
};

static imx296_reg imx296_mode_common[] = {
	{IMX296_TABLE_END, 0x00}
};

static imx296_reg imx296_mode_1440x1088_60fps[] = {
	/* capture settings */
	{IMX296_TABLE_END, 0x00}
};

enum {
	IMX296_MODE_1440x1088_60FPS,

	IMX296_MODE_COMMON,
	IMX296_START_STREAM,
	IMX296_STOP_STREAM,
};

static imx296_reg *mode_table[] = {
	[IMX296_MODE_1440x1088_60FPS] = imx296_mode_1440x1088_60fps,

	[IMX296_MODE_COMMON]  = imx296_mode_common,
	[IMX296_START_STREAM]  = imx296_start_stream,
	[IMX296_STOP_STREAM]  = imx296_stop_stream,
};

static const int imx296_60fps[] = {
	60,
};


static const struct camera_common_frmfmt imx296_frmfmt[] = {
	{{1440, 1088},	imx296_60fps, 1, 0, IMX296_MODE_1440x1088_60FPS},
};

static const struct of_device_id imx296_of_match[] = {
	{ .compatible = "nvidia,imx296", },
	{ },
};
MODULE_DEVICE_TABLE(of, imx296_of_match);

static const u32 ctrl_cid_list[] = {
	TEGRA_CAMERA_CID_GAIN,
	TEGRA_CAMERA_CID_EXPOSURE,
	TEGRA_CAMERA_CID_FRAME_RATE,
	TEGRA_CAMERA_CID_SENSOR_MODE_ID,
};

enum imx_model {
	IMX296_MODEL_MONOCHROME,
	IMX296_MODEL_COLOR,
};

/* MCLK:25MHz  1440x1080  60fps   MIPI LANE1 */
static const imx296_reg imx296_init_tab_1440_1088_60fps[] = {
	{IMX296_TABLE_END, 0x00}
};


struct imx296_mode {
	u32 sensor_mode;
	u32 sensor_depth;
	u32 sensor_ext_trig;
	u32 width;
	u32 height;
	u32 max_fps;
	const imx296_reg *reg_list;
};

struct vc_rom_table {
	char magic[12];
	char manuf[32];
	u16 manuf_id;
	char sen_manuf[8];
	char sen_type[16];
	u16 mod_id;
	u16 mod_rev;
	char regs[56];
	u16 nr_modes;
	u16 bytes_per_mode;
	char mode1[16];
	char mode2[16];
};


struct imx296 {
	struct i2c_client *i2c_client;
	struct i2c_client *rom;
	struct v4l2_subdev *subdev;
	u16 ine_integ_time;
	u32 frame_length;
	struct camera_common_data *s_data;
	struct tegracam_device *tc_dev;
	struct vc_rom_table rom_table;

	u32 sensor_ext_trig;
	enum imx_model model;

	int hflip;
	int vflip;
	u16 analog_gain;
	u16 digital_gain;
	u32 exposure_time;
};

static struct i2c_client *rom;

static const struct regmap_config sensor_regmap_config = {
	.reg_bits = 16,
	.val_bits = 8,
	.cache_type = REGCACHE_RBTREE,
	.use_single_rw = true,
};

static int reg_write(struct i2c_client *client, const u16 addr, const u8 data)
{
	struct i2c_adapter *adap = client->adapter;
	struct i2c_msg msg;
	u8 tx[3];
	int ret;

	msg.addr = client->addr;
	msg.buf = tx;
	msg.len = 3;
	msg.flags = 0;
	tx[0] = addr >> 8;
	tx[1] = addr & 0xff;
	tx[2] = data;
	ret = i2c_transfer(adap, &msg, 1);
	mdelay(2);

	return ret == 1 ? 0 : -EIO;
}

static int reg_read(struct i2c_client *client, const u16 addr)
{
	u8 buf[2] = {addr >> 8, addr & 0xff};
	int ret;
	struct i2c_msg msgs[] = {
		{
			.addr  = client->addr,
			.flags = 0,
			.len   = 2,
			.buf   = buf,
		}, {
			.addr  = client->addr,
			.flags = I2C_M_RD,
			.len   = 1,
			.buf   = buf,
		},
	};

	ret = i2c_transfer(client->adapter, msgs, ARRAY_SIZE(msgs));
	if (ret < 0) {
		dev_warn(&client->dev, "Reading register %x from %x failed\n",
			 addr, client->addr);
		return ret;
	}

	return buf[0];
}

static int reg_write_table(struct i2c_client *client,
			   const imx296_reg table[])
{
	const imx296_reg *reg;
	int ret;

	for (reg = table; reg->addr != IMX296_TABLE_END; reg++) {
		ret = reg_write(client, reg->addr, reg->val);
		if (ret < 0)
			return ret;
	}

	return 0;
}

static inline int imx296_read_reg(struct camera_common_data *s_data,
	u16 addr, u8 *val)
{
	int err = 0;
	u32 reg_val = 0;

	err = regmap_read(s_data->regmap, addr, &reg_val);
	*val = reg_val & 0xff;

	return err;
}

static inline int imx296_write_reg(struct camera_common_data *s_data,
	u16 addr, u8 val)
{
	int err = 0;

	err = regmap_write(s_data->regmap, addr, val);
	if (err)
		dev_err(s_data->dev, "%s: i2c write failed, 0x%x = %x",
			__func__, addr, val);

	return err;
}

static int imx296_write_table(struct imx296 *priv, const imx296_reg table[])
{
	return regmap_util_write_table_8(priv->s_data->regmap, table, NULL, 0,
		IMX296_TABLE_WAIT_MS, IMX296_TABLE_END);
}

static int imx296_set_group_hold(struct tegracam_device *tc_dev, bool val)
{
	/* imx296 does not support group hold */
	return 0;
}

static int imx296_set_gain(struct tegracam_device *tc_dev, s64 val)
{
	struct imx296 *priv = (struct imx296 *)tc_dev->priv;
	struct i2c_client *client = priv->i2c_client;
	int ret;
	u16 gain = 0;

	gain = val;

	if (gain < 0)
		gain = 0;
	if (gain > 480)
		gain = 480;

	priv->digital_gain = gain;

	ret = reg_write(client, 0x3205, (priv->digital_gain >> 8) & 0x01);
	ret |= reg_write(client, 0x3204, priv->digital_gain & 0xff);

	return ret;
}

static int imx296_set_frame_rate(struct tegracam_device *tc_dev, s64 val)
{
	struct camera_common_data *s_data = tc_dev->s_data;
	struct imx296 *priv = (struct imx296 *)tc_dev->priv;
	const struct sensor_mode_properties *mode =
		&s_data->sensor_props.sensor_modes[s_data->mode_prop_idx];
	u32 frame_length;

	frame_length = (u32)(mode->signal_properties.pixel_clock.val *
		(u64)mode->control_properties.framerate_factor /
		mode->image_properties.line_length / val);

	priv->frame_length = frame_length;

	return 0;
}


//
// IMX296
// 1H period 14.815us
// NumberOfLines=1118
//
#define H1PERIOD_296 242726 // (U32)(14.815 * 16384.0)
#define NRLINES_296  (1118)
#define TOFFSET_296  233636 // (U32)(14.260 * 16384.0)
#define VMAX_296     1118
#define EXPOSURE_TIME_MIN_296  29
#define EXPOSURE_TIME_MIN2_296 16504
#define EXPOSURE_TIME_MAX_296  15534389

static int imx296_exposure(struct imx296 *priv)
{
	struct i2c_client *client = priv->i2c_client;
	int ret;
	u32 exposure = 0;

	if (priv->exposure_time < EXPOSURE_TIME_MIN_296)
		priv->exposure_time = EXPOSURE_TIME_MIN_296;

	if (priv->exposure_time > EXPOSURE_TIME_MAX_296)
		priv->exposure_time = EXPOSURE_TIME_MAX_296;

	if (priv->exposure_time < EXPOSURE_TIME_MIN2_296) {
		// exposure = (NumberOfLines - exp_time / 1Hperiod +
		//toffset / 1Hperiod )
		exposure = (NRLINES_296  -  ((int)(priv->exposure_time) *
			16384 -	TOFFSET_296)/H1PERIOD_296);

		dev_info(&client->dev, "SHS = %d\n", exposure);

		ret  = reg_write(client, 0x3012, 0x00);
		ret |= reg_write(client, 0x3011, (VMAX_296 >> 8) & 0xff);
		ret |= reg_write(client, 0x3010,  VMAX_296       & 0xff);

		ret |= reg_write(client, 0x308f, (exposure >> 16) & 0x07);
		ret |= reg_write(client, 0x308e, (exposure >>  8) & 0xff);
		ret |= reg_write(client, 0x308d,  exposure        & 0xff);
	} else {
		exposure = 5 + ((int)(priv->exposure_time) * 16384 -
			TOFFSET_296)/H1PERIOD_296;

		dev_info(&client->dev, "VMAX = %d\n", exposure);

		ret  = reg_write(client, 0x308f, 0x00);
		ret |= reg_write(client, 0x308e, 0x00);
		ret |= reg_write(client, 0x308d, 0x04);

		ret |= reg_write(client, 0x3012, (exposure >> 16) & 0x07);
		ret |= reg_write(client, 0x3011, (exposure >>  8) & 0xff);
		ret |= reg_write(client, 0x3010,  exposure        & 0xff);

	}

	return ret;
}

static int imx296_set_exposure(struct tegracam_device *tc_dev, s64 val)
{
	int ret = 0;
	struct imx296 *priv = (struct imx296 *)tc_dev->priv;

	priv->exposure_time = val;

	switch (priv->model) {
	case IMX296_MODEL_MONOCHROME:
	case IMX296_MODEL_COLOR:
		ret = imx296_exposure(priv);
		break;
	}

	return 0;
}

static struct tegracam_ctrl_ops imx296_ctrl_ops = {
	.numctrls = ARRAY_SIZE(ctrl_cid_list),
	.ctrl_cid_list = ctrl_cid_list,
	.set_gain = imx296_set_gain,
	.set_exposure = imx296_set_exposure,
	.set_frame_rate = imx296_set_frame_rate,
	.set_group_hold = imx296_set_group_hold,
};

static int imx296_power_on(struct camera_common_data *s_data)
{
	int err = 0;
	struct camera_common_power_rail *pw = s_data->power;
	struct camera_common_pdata *pdata = s_data->pdata;
	struct device *dev = s_data->dev;

	dev_dbg(dev, "%s: power on\n", __func__);
	if (pdata && pdata->power_on) {
		err = pdata->power_on(pw);
		if (err)
			dev_err(dev, "%s failed.\n", __func__);
		else
			pw->state = SWITCH_ON;
		return err;
	}

	reg_write(rom, 0x0100, 2);
	reg_write(rom, 0x0102, 0);

	if (unlikely(!(pw->avdd || pw->iovdd || pw->dvdd)))
		goto skip_power_seqn;

	usleep_range(10, 20);

	if (pw->avdd) {
		err = regulator_enable(pw->avdd);
		if (err)
			goto imx296_avdd_fail;
	}

	if (pw->iovdd) {
		err = regulator_enable(pw->iovdd);
		if (err)
			goto imx296_iovdd_fail;
	}

	if (pw->dvdd) {
		err = regulator_enable(pw->dvdd);
		if (err)
			goto imx296_dvdd_fail;
	}

	usleep_range(10, 20);

skip_power_seqn:

	reg_write(rom, 0x0100, 0);

	/* Need to wait for t4 + t5 + t9 time as per the data sheet */
	/* t4 - 200us, t5 - 21.2ms, t9 - 1.2ms */
	usleep_range(23000, 23100);

	pw->state = SWITCH_ON;

	return 0;

imx296_dvdd_fail:
	regulator_disable(pw->iovdd);

imx296_iovdd_fail:
	regulator_disable(pw->avdd);

imx296_avdd_fail:
	dev_err(dev, "%s failed.\n", __func__);
	return -ENODEV;
}

static int imx296_power_off(struct camera_common_data *s_data)
{
	int err = 0;
	struct camera_common_power_rail *pw = s_data->power;
	struct camera_common_pdata *pdata = s_data->pdata;
	struct device *dev = s_data->dev;

	dev_dbg(dev, "%s: power off\n", __func__);
	if (pdata && pdata->power_off) {
		err = pdata->power_off(pw);
		if (err) {
			dev_err(dev, "%s failed.\n", __func__);
			return err;
		}
	} else {
		reg_write(rom, 0x0100, 2);

		usleep_range(10, 15);

		if (pw->dvdd)
			regulator_disable(pw->dvdd);
		if (pw->iovdd)
			regulator_disable(pw->iovdd);
		if (pw->avdd)
			regulator_disable(pw->avdd);
	}

	pw->state = SWITCH_OFF;
	return 0;
}

static int imx296_power_put(struct tegracam_device *tc_dev)
{
	struct camera_common_data *s_data = tc_dev->s_data;
	struct camera_common_power_rail *pw = s_data->power;

	if (unlikely(!pw))
		return -EFAULT;

	if (likely(pw->dvdd))
		devm_regulator_put(pw->dvdd);

	if (likely(pw->avdd))
		devm_regulator_put(pw->avdd);

	if (likely(pw->iovdd))
		devm_regulator_put(pw->iovdd);

	pw->dvdd = NULL;
	pw->avdd = NULL;
	pw->iovdd = NULL;

	return 0;
}

static int imx296_power_get(struct tegracam_device *tc_dev)
{
	struct device *dev = tc_dev->dev;
	struct camera_common_data *s_data = tc_dev->s_data;
	struct camera_common_power_rail *pw = s_data->power;
	struct camera_common_pdata *pdata = s_data->pdata;
	struct clk *parent;
	int err = 0;

	if (!pdata) {
		dev_err(dev, "pdata missing\n");
		return -EFAULT;
	}
	/* Sensor MCLK (aka. INCK) */
	if (pdata->mclk_name) {
		pw->mclk = devm_clk_get(dev, pdata->mclk_name);
		if (IS_ERR(pw->mclk)) {
			dev_err(dev, "unable to get clock %s\n",
				pdata->mclk_name);

			return PTR_ERR(pw->mclk);
		}

		if (pdata->parentclk_name) {
			parent = devm_clk_get(dev, pdata->parentclk_name);
			if (IS_ERR(parent)) {
				dev_err(dev, "unable to get parent clock %s",
					pdata->parentclk_name);
			} else
				clk_set_parent(pw->mclk, parent);
		}
	}
	/* analog 2.8v */
	if (pdata->regulators.avdd)
		err |= camera_common_regulator_get(dev,
				&pw->avdd, pdata->regulators.avdd);
	/* IO 1.8v */
	if (pdata->regulators.iovdd)
		err |= camera_common_regulator_get(dev,
				&pw->iovdd, pdata->regulators.iovdd);
	/* dig 1.2v */
	if (pdata->regulators.dvdd)
		err |= camera_common_regulator_get(dev,
				&pw->dvdd, pdata->regulators.dvdd);
	if (err) {
		dev_err(dev, "%s: unable to get regulator(s)\n", __func__);
		goto done;
	}

done:
	pw->state = SWITCH_OFF;
	return err;
}

static struct camera_common_pdata *imx296_parse_dt(
	struct tegracam_device *tc_dev)
{
	struct device *dev = tc_dev->dev;
	struct device_node *np = dev->of_node;
	struct camera_common_pdata *board_priv_pdata;
	const struct of_device_id *match;
	// struct camera_common_pdata *ret = NULL;
	int err = 0;

	if (!np)
		return NULL;

	match = of_match_device(imx296_of_match, dev);

	if (!match) {
		dev_err(dev, "Failed to find matching dt id\n");
		return NULL;
	}

	board_priv_pdata = devm_kzalloc(dev,
		sizeof(*board_priv_pdata), GFP_KERNEL);
	if (!board_priv_pdata)
		return NULL;

	err = of_property_read_string(np, "mclk", &board_priv_pdata->mclk_name);
	if (err)
		dev_dbg(dev,
			"mclk name not present, assume sensor driven externally\n");

	err = of_property_read_string(np, "avdd-reg",
		&board_priv_pdata->regulators.avdd);
	err |= of_property_read_string(np, "iovdd-reg",
		&board_priv_pdata->regulators.iovdd);
	err |= of_property_read_string(np, "dvdd-reg",
		&board_priv_pdata->regulators.dvdd);
	if (err)
		dev_dbg(dev,
			"avdd, iovdd and/or dvdd reglrs. not present, assume sensor powered independently\n");

	board_priv_pdata->has_eeprom =
		of_property_read_bool(np, "has-eeprom");

	return board_priv_pdata;
}

static int imx296_set_mode(struct tegracam_device *tc_dev)
{
	struct imx296 *priv = (struct imx296 *)tegracam_get_privdata(tc_dev);
	struct camera_common_data *s_data = tc_dev->s_data;
	int err = 0;

	err = imx296_write_table(priv, mode_table[IMX296_MODE_COMMON]);
	if (err)
		return err;

	err = imx296_write_table(priv, mode_table[s_data->mode]);
	if (err)
		return err;

	return 0;
}

static int imx296_start_streaming(struct tegracam_device *tc_dev)
{
	struct imx296 *priv = (struct imx296 *)tegracam_get_privdata(tc_dev);
	struct i2c_client *client = priv->i2c_client;
	int ret;

	ret = reg_write_table(client, imx296_init_tab_1440_1088_60fps);
	if (ret)
		return ret;

	if (priv->sensor_ext_trig) {
		u64 exposure = (priv->exposure_time * 10000) / 185185;

		int addr = 0x0108; // ext trig enable
		int data = 1; // external trigger enable

		ret = reg_write(priv->rom, addr, data);
#if 0
		// TODO
		addr = 0x0103; // flash output enable
		data =      1; // flash output enable
		ret = reg_write(priv->rom, addr, data);
#endif
		addr = 0x0109; // shutter lsb
		data = exposure & 0xff;
		ret = reg_write(priv->rom, addr, data);

		addr = 0x010a;
		data = (exposure >> 8) & 0xff;
		ret = reg_write(priv->rom, addr, data);

		addr = 0x010b;
		data = (exposure >> 16) & 0xff;
		ret = reg_write(priv->rom, addr, data);

		addr = 0x010c; // shutter msb
		data = (exposure >> 24) & 0xff;
		ret = reg_write(priv->rom, addr, data);

		usleep_range(5000, 5100);

		return reg_write_table(client, imx296_start_stream);

	} else {
		int addr = 0x0108; // ext trig enable
		int data = 0; // external trigger disable

		ret = reg_write(priv->rom, addr, data);
		usleep_range(5000, 5100);

		return reg_write_table(client, imx296_start_stream);

	}

}

static int imx296_stop_streaming(struct tegracam_device *tc_dev)
{
	int err = 0;
	struct imx296 *priv = (struct imx296 *)tegracam_get_privdata(tc_dev);

	err = imx296_write_table(priv, mode_table[IMX296_STOP_STREAM]);

	usleep_range(50000, 51000);

	return err;
}

static struct camera_common_sensor_ops imx296_common_ops = {
	.numfrmfmts = ARRAY_SIZE(imx296_frmfmt),
	.frmfmt_table = imx296_frmfmt,
	.power_on = imx296_power_on,
	.power_off = imx296_power_off,
	.write_reg = imx296_write_reg,
	.read_reg = imx296_read_reg,
	.parse_dt = imx296_parse_dt,
	.power_get = imx296_power_get,
	.power_put = imx296_power_put,
	.set_mode = imx296_set_mode,
	.start_streaming = imx296_start_streaming,
	.stop_streaming = imx296_stop_streaming,
};

static int imx296_board_setup(struct imx296 *priv)
{
	struct camera_common_data *s_data = priv->s_data;
	struct camera_common_pdata *pdata = s_data->pdata;
	struct device *dev = s_data->dev;
	int err = 0;

	if (pdata->mclk_name) {
		err = camera_common_mclk_enable(s_data);
		if (err) {
			dev_err(dev, "error turning on mclk (%d)\n", err);
			goto done;
		}
	}

	err = imx296_power_on(s_data);
	if (err) {
		dev_err(dev, "error during power on sensor (%d)\n", err);
		goto err_power_on;
	}

	imx296_power_off(s_data);

err_power_on:
	if (pdata->mclk_name)
		camera_common_mclk_disable(s_data);

done:
	return err;
}

static int imx296_open(struct v4l2_subdev *sd, struct v4l2_subdev_fh *fh)
{
	struct i2c_client *client = v4l2_get_subdevdata(sd);

	dev_dbg(&client->dev, "%s:\n", __func__);

	return 0;
}

static const struct v4l2_subdev_internal_ops imx296_subdev_internal_ops = {
	.open = imx296_open,
};


static struct i2c_client *imx296_probe_vc_rom(
	struct i2c_adapter *adapter,
	u8 addr)
{
	struct i2c_board_info info = {
		I2C_BOARD_INFO("dummy", addr),
	};
	unsigned short addr_list[2] = { addr, I2C_CLIENT_END };

	return i2c_new_probed_device(adapter, &info, addr_list, NULL);
}

static int imx296_probe(struct i2c_client *client,
	const struct i2c_device_id *id)
{
	struct device *dev = &client->dev;
	struct tegracam_device *tc_dev;
	struct imx296 *priv;
	int err;
	struct i2c_adapter *adapter = to_i2c_adapter(client->dev.parent);
	int addr, reg, data;

	dev_dbg(dev, "probing v4l2 sensor at addr 0x%0x\n", client->addr);

	if (!IS_ENABLED(CONFIG_OF) || !client->dev.of_node)
		return -EINVAL;

	priv = devm_kzalloc(dev,
			sizeof(struct imx296), GFP_KERNEL);
	if (!priv)
		return -ENOMEM;

	tc_dev = devm_kzalloc(dev,
			sizeof(struct tegracam_device), GFP_KERNEL);
	if (!tc_dev)
		return -ENOMEM;

	if (!i2c_check_functionality(adapter, I2C_FUNC_SMBUS_BYTE_DATA)) {
		dev_warn(&adapter->dev,
			 "I2C-Adapter doesn't support I2C_FUNC_SMBUS_BYTE\n");
		return -EIO;
	}


	priv->rom = imx296_probe_vc_rom(adapter, 0x10);

	if (priv->rom) {
		static int i = 1;

		rom = priv->rom;

#if 1
		for (addr = 0; addr < sizeof(priv->rom_table); addr++) {
			reg = reg_read(priv->rom, addr+0x1000);
			if (reg < 0) {
				i2c_unregister_device(priv->rom);
				return -EIO;
			}
			*((char *)(&(priv->rom_table))+addr) = (char)reg;
			dev_dbg(&client->dev,
				"addr=0x%04x reg=0x%02x\n",
				addr+0x1000,
				reg);
		}

		dev_info(&client->dev, "VC FPGA found!\n");

		dev_info(&client->dev, "[ MAGIC  ] [ %s ]\n",
				priv->rom_table.magic);

		dev_info(&client->dev, "[ MANUF. ] [ %s ] [ MID=0x%04x ]\n",
				priv->rom_table.manuf,
				priv->rom_table.manuf_id);

		dev_info(&client->dev, "[ SENSOR ] [ %s %s ]\n",
				priv->rom_table.sen_manuf,
				priv->rom_table.sen_type);

		dev_info(&client->dev, "[ MODULE ] [ ID=0x%04x ] [ REV=0x%04x ]\n",
				priv->rom_table.mod_id,
				priv->rom_table.mod_rev);

		dev_info(&client->dev, "[ MODES  ] [ NR=0x%04x ] [ BPM=0x%04x ]\n",
				priv->rom_table.nr_modes,
				priv->rom_table.bytes_per_mode);

		if (priv->rom_table.sen_type) {
			u32 len = strnlen(priv->rom_table.sen_type, 16);

			if (len > 0 && len < 17) {
				if (*(priv->rom_table.sen_type+len-1) == 'C') {
					dev_info(&client->dev,
					"[ COLOR  ] [  %c ]\n",
					*(priv->rom_table.sen_type+len-1));

					// DEFAULT IMX296
					priv->model = IMX296_MODEL_COLOR;
				} else {
					dev_info(&client->dev, "[ MONO   ] [ B/W ]\n");

					// DEFAULT IMX296
					priv->model = IMX296_MODEL_MONOCHROME;
				}
			} //TODO else
		}
#endif
		addr = 0x0100; // reset
		data = 2; // powerdown sensor
		reg = reg_write(priv->rom, addr, data);

		addr = 0x0102; // mode
		data = sensor_mode; // default 10-bit streaming
		reg = reg_write(priv->rom, addr, data);

		addr = 0x0100; // reset
		data =      0; // powerup sensor
		reg = reg_write(priv->rom, addr, data);

		while (1) {
			mdelay(100); // wait 100ms

			addr = 0x0101; // status
			reg = reg_read(priv->rom, addr);

			if (reg & 0x80)
				break;

			if (reg & 0x01)
				dev_err(&client->dev, "!!! ERROR !!! setting VC Sensor MODE=%d STATUS=0x%02x i=%d\n",
					sensor_mode, reg, i);

			if (i++ > 4)
				break;
		}

		dev_info(&client->dev, "VC Sensor MODE=%d PowerOn STATUS=0x%02x i=%d\n",
			sensor_mode, reg, i);

	} else {
		dev_err(&client->dev, "Error !!! VC FPGA not found !!!\n");
		return -EIO;
	}

	priv->i2c_client = tc_dev->client = client;
	tc_dev->dev = dev;
	strncpy(tc_dev->name, "imx296", sizeof(tc_dev->name));
	tc_dev->dev_regmap_config = &sensor_regmap_config;
	tc_dev->sensor_ops = &imx296_common_ops;
	tc_dev->v4l2sd_internal_ops = &imx296_subdev_internal_ops;
	tc_dev->tcctrl_ops = &imx296_ctrl_ops;

	err = tegracam_device_register(tc_dev);
	if (err) {
		dev_err(dev, "tegra camera driver registration failed\n");
		return err;
	}

	priv->tc_dev = tc_dev;
	priv->s_data = tc_dev->s_data;
	priv->subdev = &tc_dev->s_data->subdev;

	tegracam_set_privdata(tc_dev, (void *)priv);

	err = imx296_board_setup(priv);
	if (err) {
		dev_err(dev, "board setup failed\n");
		return err;
	}

	err = tegracam_v4l2subdev_register(tc_dev, true);
	if (err) {
		dev_err(dev, "tegra camera subdev registration failed\n");
		return err;
	}
	dev_dbg(dev, "detected imx296 sensor\n");
	return 0;
}

static int imx296_remove(struct i2c_client *client)
{
	struct camera_common_data *s_data = to_camera_common_data(&client->dev);
	struct imx296 *priv = (struct imx296 *)s_data->priv;

	tegracam_v4l2subdev_unregister(priv->tc_dev);
	tegracam_device_unregister(priv->tc_dev);

	return 0;
}

static const struct i2c_device_id imx296_id[] = {
	{ "imx296", 0 },
	{ }
};
MODULE_DEVICE_TABLE(i2c, imx296_id);

static struct i2c_driver imx296_i2c_driver = {
	.driver = {
		.name = "imx296",
		.owner = THIS_MODULE,
		.of_match_table = of_match_ptr(imx296_of_match),
	},
	.probe = imx296_probe,
	.remove = imx296_remove,
	.id_table = imx296_id,
};
module_i2c_driver(imx296_i2c_driver);

MODULE_DESCRIPTION("Media Controller driver for Sony IMX296");
MODULE_AUTHOR("Andrey Pahomov <pahomov.and@gmail.com>");
MODULE_LICENSE("GPL v2");
