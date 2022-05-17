
#include "../native/builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)


using namespace wp;


static CUDA_CALLABLE vec3 quat_decompose(quat var_q)
{
        //---------
    // primal vars
    const float32 var_0 = 1.0;
    const float32 var_1 = 0.0;
    vec3 var_2;
    vec3 var_3;
    vec3 var_4;
    vec3 var_5;
    vec3 var_6;
    vec3 var_7;
    mat33 var_8;
    const int32 var_9 = 1;
    const int32 var_10 = 2;
    float32 var_11;
    float32 var_12;
    float32 var_13;
    const int32 var_14 = 0;
    float32 var_15;
    float32 var_16;
    float32 var_17;
    float32 var_18;
    float32 var_19;
    float32 var_20;
    vec3 var_21;
    vec3 var_22;
    //---------
    // forward
    var_2 = wp::vec3(var_0, var_1, var_1);
    var_3 = wp::quat_rotate(var_q, var_2);
    var_4 = wp::vec3(var_1, var_0, var_1);
    var_5 = wp::quat_rotate(var_q, var_4);
    var_6 = wp::vec3(var_1, var_1, var_0);
    var_7 = wp::quat_rotate(var_q, var_6);
    var_8 = wp::mat33(var_3, var_5, var_7);
    var_11 = wp::index(var_8, var_9, var_10);
    var_12 = wp::index(var_8, var_10, var_10);
    var_13 = wp::atan2(var_11, var_12);
    var_15 = wp::index(var_8, var_14, var_10);
    var_16 = wp::neg(var_15);
    var_17 = wp::asin(var_16);
    var_18 = wp::index(var_8, var_14, var_9);
    var_19 = wp::index(var_8, var_14, var_14);
    var_20 = wp::atan2(var_18, var_19);
    var_21 = wp::vec3(var_13, var_17, var_20);
    var_22 = wp::neg(var_21);
    return var_22;

}

static CUDA_CALLABLE void adj_quat_decompose(quat var_q,
	quat & adj_q,
	vec3 & adj_ret)
{
        //---------
    // primal vars
    const float32 var_0 = 1.0;
    const float32 var_1 = 0.0;
    vec3 var_2;
    vec3 var_3;
    vec3 var_4;
    vec3 var_5;
    vec3 var_6;
    vec3 var_7;
    mat33 var_8;
    const int32 var_9 = 1;
    const int32 var_10 = 2;
    float32 var_11;
    float32 var_12;
    float32 var_13;
    const int32 var_14 = 0;
    float32 var_15;
    float32 var_16;
    float32 var_17;
    float32 var_18;
    float32 var_19;
    float32 var_20;
    vec3 var_21;
    vec3 var_22;
    //---------
    // dual vars
    float32 adj_0 = 0;
    float32 adj_1 = 0;
    vec3 adj_2 = 0;
    vec3 adj_3 = 0;
    vec3 adj_4 = 0;
    vec3 adj_5 = 0;
    vec3 adj_6 = 0;
    vec3 adj_7 = 0;
    mat33 adj_8 = 0;
    int32 adj_9 = 0;
    int32 adj_10 = 0;
    float32 adj_11 = 0;
    float32 adj_12 = 0;
    float32 adj_13 = 0;
    int32 adj_14 = 0;
    float32 adj_15 = 0;
    float32 adj_16 = 0;
    float32 adj_17 = 0;
    float32 adj_18 = 0;
    float32 adj_19 = 0;
    float32 adj_20 = 0;
    vec3 adj_21 = 0;
    vec3 adj_22 = 0;
    //---------
    // forward
    var_2 = wp::vec3(var_0, var_1, var_1);
    var_3 = wp::quat_rotate(var_q, var_2);
    var_4 = wp::vec3(var_1, var_0, var_1);
    var_5 = wp::quat_rotate(var_q, var_4);
    var_6 = wp::vec3(var_1, var_1, var_0);
    var_7 = wp::quat_rotate(var_q, var_6);
    var_8 = wp::mat33(var_3, var_5, var_7);
    var_11 = wp::index(var_8, var_9, var_10);
    var_12 = wp::index(var_8, var_10, var_10);
    var_13 = wp::atan2(var_11, var_12);
    var_15 = wp::index(var_8, var_14, var_10);
    var_16 = wp::neg(var_15);
    var_17 = wp::asin(var_16);
    var_18 = wp::index(var_8, var_14, var_9);
    var_19 = wp::index(var_8, var_14, var_14);
    var_20 = wp::atan2(var_18, var_19);
    var_21 = wp::vec3(var_13, var_17, var_20);
    var_22 = wp::neg(var_21);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_22 += adj_ret;
    wp::adj_neg(var_21, adj_21, adj_22);
    wp::adj_vec3(var_13, var_17, var_20, adj_13, adj_17, adj_20, adj_21);
    wp::adj_atan2(var_18, var_19, adj_18, adj_19, adj_20);
    wp::adj_index(var_8, var_14, var_14, adj_8, adj_14, adj_14, adj_19);
    wp::adj_index(var_8, var_14, var_9, adj_8, adj_14, adj_9, adj_18);
    wp::adj_asin(var_16, adj_16, adj_17);
    wp::adj_neg(var_15, adj_15, adj_16);
    wp::adj_index(var_8, var_14, var_10, adj_8, adj_14, adj_10, adj_15);
    wp::adj_atan2(var_11, var_12, adj_11, adj_12, adj_13);
    wp::adj_index(var_8, var_10, var_10, adj_8, adj_10, adj_10, adj_12);
    wp::adj_index(var_8, var_9, var_10, adj_8, adj_9, adj_10, adj_11);
    wp::adj_mat33(var_3, var_5, var_7, adj_3, adj_5, adj_7, adj_8);
    wp::adj_quat_rotate(var_q, var_6, adj_q, adj_6, adj_7);
    wp::adj_vec3(var_1, var_1, var_0, adj_1, adj_1, adj_0, adj_6);
    wp::adj_quat_rotate(var_q, var_4, adj_q, adj_4, adj_5);
    wp::adj_vec3(var_1, var_0, var_1, adj_1, adj_0, adj_1, adj_4);
    wp::adj_quat_rotate(var_q, var_2, adj_q, adj_2, adj_3);
    wp::adj_vec3(var_0, var_1, var_1, adj_0, adj_1, adj_1, adj_2);
    return;

}


static CUDA_CALLABLE quat quat_twist(vec3 var_axis,
	quat var_q)
{
        //---------
    // primal vars
    const int32 var_0 = 0;
    float32 var_1;
    const int32 var_2 = 1;
    float32 var_3;
    const int32 var_4 = 2;
    float32 var_5;
    vec3 var_6;
    float32 var_7;
    vec3 var_8;
    float32 var_9;
    float32 var_10;
    float32 var_11;
    const int32 var_12 = 3;
    float32 var_13;
    quat var_14;
    quat var_15;
    //---------
    // forward
    var_1 = wp::index(var_q, var_0);
    var_3 = wp::index(var_q, var_2);
    var_5 = wp::index(var_q, var_4);
    var_6 = wp::vec3(var_1, var_3, var_5);
    var_7 = wp::dot(var_6, var_axis);
    var_8 = wp::mul(var_7, var_axis);
    var_9 = wp::index(var_8, var_0);
    var_10 = wp::index(var_8, var_2);
    var_11 = wp::index(var_8, var_4);
    var_13 = wp::index(var_q, var_12);
    var_14 = wp::quat(var_9, var_10, var_11, var_13);
    var_15 = wp::normalize(var_14);
    return var_15;

}

static CUDA_CALLABLE void adj_quat_twist(vec3 var_axis,
	quat var_q,
	vec3 & adj_axis,
	quat & adj_q,
	quat & adj_ret)
{
        //---------
    // primal vars
    const int32 var_0 = 0;
    float32 var_1;
    const int32 var_2 = 1;
    float32 var_3;
    const int32 var_4 = 2;
    float32 var_5;
    vec3 var_6;
    float32 var_7;
    vec3 var_8;
    float32 var_9;
    float32 var_10;
    float32 var_11;
    const int32 var_12 = 3;
    float32 var_13;
    quat var_14;
    quat var_15;
    //---------
    // dual vars
    int32 adj_0 = 0;
    float32 adj_1 = 0;
    int32 adj_2 = 0;
    float32 adj_3 = 0;
    int32 adj_4 = 0;
    float32 adj_5 = 0;
    vec3 adj_6 = 0;
    float32 adj_7 = 0;
    vec3 adj_8 = 0;
    float32 adj_9 = 0;
    float32 adj_10 = 0;
    float32 adj_11 = 0;
    int32 adj_12 = 0;
    float32 adj_13 = 0;
    quat adj_14 = 0;
    quat adj_15 = 0;
    //---------
    // forward
    var_1 = wp::index(var_q, var_0);
    var_3 = wp::index(var_q, var_2);
    var_5 = wp::index(var_q, var_4);
    var_6 = wp::vec3(var_1, var_3, var_5);
    var_7 = wp::dot(var_6, var_axis);
    var_8 = wp::mul(var_7, var_axis);
    var_9 = wp::index(var_8, var_0);
    var_10 = wp::index(var_8, var_2);
    var_11 = wp::index(var_8, var_4);
    var_13 = wp::index(var_q, var_12);
    var_14 = wp::quat(var_9, var_10, var_11, var_13);
    var_15 = wp::normalize(var_14);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_15 += adj_ret;
    wp::adj_normalize(var_14, adj_14, adj_15);
    wp::adj_quat(var_9, var_10, var_11, var_13, adj_9, adj_10, adj_11, adj_13, adj_14);
    wp::adj_index(var_q, var_12, adj_q, adj_12, adj_13);
    wp::adj_index(var_8, var_4, adj_8, adj_4, adj_11);
    wp::adj_index(var_8, var_2, adj_8, adj_2, adj_10);
    wp::adj_index(var_8, var_0, adj_8, adj_0, adj_9);
    wp::adj_mul(var_7, var_axis, adj_7, adj_axis, adj_8);
    wp::adj_dot(var_6, var_axis, adj_6, adj_axis, adj_7);
    wp::adj_vec3(var_1, var_3, var_5, adj_1, adj_3, adj_5, adj_6);
    wp::adj_index(var_q, var_4, adj_q, adj_4, adj_5);
    wp::adj_index(var_q, var_2, adj_q, adj_2, adj_3);
    wp::adj_index(var_q, var_0, adj_q, adj_0, adj_1);
    return;

}



extern "C" __global__ void eval_articulation_fk_cuda_kernel_forward(launch_bounds_t dim,
	array_t<int32> var_articulation_start,
	array_t<int32> var_articulation_mask,
	array_t<float32> var_joint_q,
	array_t<float32> var_joint_qd,
	array_t<int32> var_joint_q_start,
	array_t<int32> var_joint_qd_start,
	array_t<int32> var_joint_type,
	array_t<int32> var_joint_parent,
	array_t<transform> var_joint_X_p,
	array_t<transform> var_joint_X_c,
	array_t<vec3> var_joint_axis,
	array_t<vec3> var_body_com,
	array_t<transform> var_body_q,
	array_t<spatial_vector> var_body_qd)
{
    int _idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (_idx >= dim.size) 
        return;

    set_launch_bounds(dim);

        //---------
    // primal vars
    int32 var_0;
    int32 var_1;
    const int32 var_2 = 0;
    bool var_3;
    int32 var_4;
    const int32 var_5 = 1;
    int32 var_6;
    int32 var_7;
    range_t var_8;
    int32 var_9;
    int32 var_10;
    transform var_11;
    spatial_vector var_12;
    bool var_13;
    transform var_14;
    spatial_vector var_15;
    transform var_16;
    spatial_vector var_17;
    int32 var_18;
    vec3 var_19;
    transform var_20;
    transform var_21;
    int32 var_22;
    int32 var_23;
    const int32 var_24 = 0;
    bool var_25;
    float32 var_26;
    float32 var_27;
    vec3 var_28;
    quat var_29;
    transform var_30;
    vec3 var_31;
    vec3 var_32;
    spatial_vector var_33;
    const int32 var_34 = 1;
    bool var_35;
    float32 var_36;
    float32 var_37;
    vec3 var_38;
    quat var_39;
    transform var_40;
    vec3 var_41;
    vec3 var_42;
    spatial_vector var_43;
    float32 var_44;
    float32 var_45;
    transform var_46;
    spatial_vector var_47;
    const int32 var_48 = 2;
    bool var_49;
    int32 var_50;
    float32 var_51;
    int32 var_52;
    float32 var_53;
    const int32 var_54 = 2;
    int32 var_55;
    float32 var_56;
    const int32 var_57 = 3;
    int32 var_58;
    float32 var_59;
    quat var_60;
    int32 var_61;
    float32 var_62;
    int32 var_63;
    float32 var_64;
    int32 var_65;
    float32 var_66;
    vec3 var_67;
    vec3 var_68;
    transform var_69;
    vec3 var_70;
    spatial_vector var_71;
    transform var_72;
    spatial_vector var_73;
    const int32 var_74 = 3;
    bool var_75;
    transform var_76;
    vec3 var_77;
    vec3 var_78;
    spatial_vector var_79;
    transform var_80;
    spatial_vector var_81;
    const int32 var_82 = 4;
    bool var_83;
    int32 var_84;
    float32 var_85;
    int32 var_86;
    float32 var_87;
    int32 var_88;
    float32 var_89;
    vec3 var_90;
    int32 var_91;
    float32 var_92;
    const int32 var_93 = 4;
    int32 var_94;
    float32 var_95;
    const int32 var_96 = 5;
    int32 var_97;
    float32 var_98;
    const int32 var_99 = 6;
    int32 var_100;
    float32 var_101;
    quat var_102;
    transform var_103;
    int32 var_104;
    float32 var_105;
    int32 var_106;
    float32 var_107;
    int32 var_108;
    float32 var_109;
    vec3 var_110;
    int32 var_111;
    float32 var_112;
    int32 var_113;
    float32 var_114;
    int32 var_115;
    float32 var_116;
    vec3 var_117;
    spatial_vector var_118;
    transform var_119;
    spatial_vector var_120;
    transform var_121;
    spatial_vector var_122;
    const int32 var_123 = 5;
    bool var_124;
    quat var_125;
    const float32 var_126 = 1.0;
    const float32 var_127 = 0.0;
    vec3 var_128;
    vec3 var_129;
    vec3 var_130;
    vec3 var_131;
    vec3 var_132;
    vec3 var_133;
    vec3 var_134;
    int32 var_135;
    float32 var_136;
    quat var_137;
    vec3 var_138;
    int32 var_139;
    float32 var_140;
    quat var_141;
    quat var_142;
    vec3 var_143;
    int32 var_144;
    float32 var_145;
    quat var_146;
    vec3 var_147;
    quat var_148;
    quat var_149;
    transform var_150;
    int32 var_151;
    float32 var_152;
    vec3 var_153;
    int32 var_154;
    float32 var_155;
    vec3 var_156;
    vec3 var_157;
    int32 var_158;
    float32 var_159;
    vec3 var_160;
    vec3 var_161;
    vec3 var_162;
    spatial_vector var_163;
    transform var_164;
    spatial_vector var_165;
    transform var_166;
    spatial_vector var_167;
    transform var_168;
    spatial_vector var_169;
    const int32 var_170 = 6;
    bool var_171;
    quat var_172;
    vec3 var_173;
    vec3 var_174;
    vec3 var_175;
    vec3 var_176;
    vec3 var_177;
    int32 var_178;
    float32 var_179;
    quat var_180;
    vec3 var_181;
    int32 var_182;
    float32 var_183;
    quat var_184;
    vec3 var_185;
    quat var_186;
    transform var_187;
    int32 var_188;
    float32 var_189;
    vec3 var_190;
    int32 var_191;
    float32 var_192;
    vec3 var_193;
    vec3 var_194;
    vec3 var_195;
    spatial_vector var_196;
    transform var_197;
    spatial_vector var_198;
    transform var_199;
    spatial_vector var_200;
    transform var_201;
    spatial_vector var_202;
    quat var_203;
    vec3 var_204;
    vec3 var_205;
    vec3 var_206;
    quat var_207;
    vec3 var_208;
    quat var_209;
    transform var_210;
    transform var_211;
    vec3 var_212;
    vec3 var_213;
    vec3 var_214;
    vec3 var_215;
    vec3 var_216;
    vec3 var_217;
    vec3 var_218;
    spatial_vector var_219;
    spatial_vector var_220;
    //---------
    // forward
        var_0 = wp::tid();
        if (var_articulation_mask) {
        	var_1 = wp::load(var_articulation_mask, var_0);
        	var_3 = (var_1 == var_2);
        	if (var_3) {
        		return;
        	}
        }
        var_4 = wp::load(var_articulation_start, var_0);
        var_6 = wp::add(var_0, var_5);
        var_7 = wp::load(var_articulation_start, var_6);
        var_8 = wp::range(var_4, var_7);
        for_start_1:;
        	if (iter_cmp(var_8) == 0) goto for_end_1;
        	var_9 = wp::iter_next(var_8);
        	var_10 = wp::load(var_joint_parent, var_9);
        	var_11 = wp::transform_identity();
        	var_12 = wp::spatial_vector();
        	var_13 = (var_10 >= var_2);
        	if (var_13) {
        		var_14 = wp::load(var_body_q, var_10);
        		var_15 = wp::load(var_body_qd, var_10);
        	}
        	var_16 = wp::select(var_13, var_11, var_14);
        	var_17 = wp::select(var_13, var_12, var_15);
        	var_18 = wp::load(var_joint_type, var_9);
        	var_19 = wp::load(var_joint_axis, var_9);
        	var_20 = wp::load(var_joint_X_p, var_9);
        	var_21 = wp::load(var_joint_X_c, var_9);
        	var_22 = wp::load(var_joint_q_start, var_9);
        	var_23 = wp::load(var_joint_qd_start, var_9);
        	var_25 = (var_18 == var_24);
        	if (var_25) {
        		var_26 = wp::load(var_joint_q, var_22);
        		var_27 = wp::load(var_joint_qd, var_23);
        		var_28 = wp::mul(var_19, var_26);
        		var_29 = wp::quat_identity();
        		var_30 = wp::transform(var_28, var_29);
        		var_31 = wp::vec3();
        		var_32 = wp::mul(var_19, var_27);
        		var_33 = wp::spatial_vector(var_31, var_32);
        	}
        	var_35 = (var_18 == var_34);
        	if (var_35) {
        		var_36 = wp::load(var_joint_q, var_22);
        		var_37 = wp::load(var_joint_qd, var_23);
        		var_38 = wp::vec3();
        		var_39 = wp::quat_from_axis_angle(var_19, var_36);
        		var_40 = wp::transform(var_38, var_39);
        		var_41 = wp::mul(var_19, var_37);
        		var_42 = wp::vec3();
        		var_43 = wp::spatial_vector(var_41, var_42);
        	}
        	var_44 = wp::select(var_35, var_26, var_36);
        	var_45 = wp::select(var_35, var_27, var_37);
        	var_46 = wp::select(var_35, var_30, var_40);
        	var_47 = wp::select(var_35, var_33, var_43);
        	var_49 = (var_18 == var_48);
        	if (var_49) {
        		var_50 = wp::add(var_22, var_2);
        		var_51 = wp::load(var_joint_q, var_50);
        		var_52 = wp::add(var_22, var_5);
        		var_53 = wp::load(var_joint_q, var_52);
        		var_55 = wp::add(var_22, var_54);
        		var_56 = wp::load(var_joint_q, var_55);
        		var_58 = wp::add(var_22, var_57);
        		var_59 = wp::load(var_joint_q, var_58);
        		var_60 = wp::quat(var_51, var_53, var_56, var_59);
        		var_61 = wp::add(var_23, var_2);
        		var_62 = wp::load(var_joint_qd, var_61);
        		var_63 = wp::add(var_23, var_5);
        		var_64 = wp::load(var_joint_qd, var_63);
        		var_65 = wp::add(var_23, var_54);
        		var_66 = wp::load(var_joint_qd, var_65);
        		var_67 = wp::vec3(var_62, var_64, var_66);
        		var_68 = wp::vec3();
        		var_69 = wp::transform(var_68, var_60);
        		var_70 = wp::vec3();
        		var_71 = wp::spatial_vector(var_67, var_70);
        	}
        	var_72 = wp::select(var_49, var_46, var_69);
        	var_73 = wp::select(var_49, var_47, var_71);
        	var_75 = (var_18 == var_74);
        	if (var_75) {
        		var_76 = wp::transform_identity();
        		var_77 = wp::vec3();
        		var_78 = wp::vec3();
        		var_79 = wp::spatial_vector(var_77, var_78);
        	}
        	var_80 = wp::select(var_75, var_72, var_76);
        	var_81 = wp::select(var_75, var_73, var_79);
        	var_83 = (var_18 == var_82);
        	if (var_83) {
        		var_84 = wp::add(var_22, var_2);
        		var_85 = wp::load(var_joint_q, var_84);
        		var_86 = wp::add(var_22, var_5);
        		var_87 = wp::load(var_joint_q, var_86);
        		var_88 = wp::add(var_22, var_54);
        		var_89 = wp::load(var_joint_q, var_88);
        		var_90 = wp::vec3(var_85, var_87, var_89);
        		var_91 = wp::add(var_22, var_57);
        		var_92 = wp::load(var_joint_q, var_91);
        		var_94 = wp::add(var_22, var_93);
        		var_95 = wp::load(var_joint_q, var_94);
        		var_97 = wp::add(var_22, var_96);
        		var_98 = wp::load(var_joint_q, var_97);
        		var_100 = wp::add(var_22, var_99);
        		var_101 = wp::load(var_joint_q, var_100);
        		var_102 = wp::quat(var_92, var_95, var_98, var_101);
        		var_103 = wp::transform(var_90, var_102);
        		var_104 = wp::add(var_23, var_2);
        		var_105 = wp::load(var_joint_qd, var_104);
        		var_106 = wp::add(var_23, var_5);
        		var_107 = wp::load(var_joint_qd, var_106);
        		var_108 = wp::add(var_23, var_54);
        		var_109 = wp::load(var_joint_qd, var_108);
        		var_110 = wp::vec3(var_105, var_107, var_109);
        		var_111 = wp::add(var_23, var_57);
        		var_112 = wp::load(var_joint_qd, var_111);
        		var_113 = wp::add(var_23, var_93);
        		var_114 = wp::load(var_joint_qd, var_113);
        		var_115 = wp::add(var_23, var_96);
        		var_116 = wp::load(var_joint_qd, var_115);
        		var_117 = wp::vec3(var_112, var_114, var_116);
        		var_118 = wp::spatial_vector(var_110, var_117);
        		wp::copy(var_119, var_103);
        		wp::copy(var_120, var_118);
        	}
        	var_121 = wp::select(var_83, var_80, var_119);
        	var_122 = wp::select(var_83, var_81, var_120);
        	var_124 = (var_18 == var_123);
        	if (var_124) {
        		var_125 = wp::transform_get_rotation(var_21);
        		var_128 = wp::vec3(var_126, var_127, var_127);
        		var_129 = wp::quat_rotate(var_125, var_128);
        		var_130 = wp::vec3(var_127, var_126, var_127);
        		var_131 = wp::quat_rotate(var_125, var_130);
        		var_132 = wp::vec3(var_127, var_127, var_126);
        		var_133 = wp::quat_rotate(var_125, var_132);
        		wp::copy(var_134, var_129);
        		var_135 = wp::add(var_22, var_2);
        		var_136 = wp::load(var_joint_q, var_135);
        		var_137 = wp::quat_from_axis_angle(var_134, var_136);
        		var_138 = wp::quat_rotate(var_137, var_131);
        		var_139 = wp::add(var_22, var_5);
        		var_140 = wp::load(var_joint_q, var_139);
        		var_141 = wp::quat_from_axis_angle(var_138, var_140);
        		var_142 = wp::mul(var_141, var_137);
        		var_143 = wp::quat_rotate(var_142, var_133);
        		var_144 = wp::add(var_22, var_54);
        		var_145 = wp::load(var_joint_q, var_144);
        		var_146 = wp::quat_from_axis_angle(var_143, var_145);
        		var_147 = wp::vec3();
        		var_148 = wp::mul(var_146, var_141);
        		var_149 = wp::mul(var_148, var_137);
        		var_150 = wp::transform(var_147, var_149);
        		var_151 = wp::add(var_23, var_2);
        		var_152 = wp::load(var_joint_qd, var_151);
        		var_153 = wp::mul(var_134, var_152);
        		var_154 = wp::add(var_23, var_5);
        		var_155 = wp::load(var_joint_qd, var_154);
        		var_156 = wp::mul(var_138, var_155);
        		var_157 = wp::add(var_153, var_156);
        		var_158 = wp::add(var_23, var_54);
        		var_159 = wp::load(var_joint_qd, var_158);
        		var_160 = wp::mul(var_143, var_159);
        		var_161 = wp::add(var_157, var_160);
        		var_162 = wp::vec3();
        		var_163 = wp::spatial_vector(var_161, var_162);
        		wp::copy(var_164, var_150);
        		wp::copy(var_165, var_163);
        	}
        	var_166 = wp::select(var_124, var_121, var_164);
        	var_167 = wp::select(var_124, var_122, var_165);
        	var_168 = wp::select(var_124, var_103, var_150);
        	var_169 = wp::select(var_124, var_118, var_163);
        	var_171 = (var_18 == var_170);
        	if (var_171) {
        		var_172 = wp::transform_get_rotation(var_21);
        		var_173 = wp::vec3(var_126, var_127, var_127);
        		var_174 = wp::quat_rotate(var_172, var_173);
        		var_175 = wp::vec3(var_127, var_126, var_127);
        		var_176 = wp::quat_rotate(var_172, var_175);
        		wp::copy(var_177, var_174);
        		var_178 = wp::add(var_22, var_2);
        		var_179 = wp::load(var_joint_q, var_178);
        		var_180 = wp::quat_from_axis_angle(var_177, var_179);
        		var_181 = wp::quat_rotate(var_180, var_176);
        		var_182 = wp::add(var_22, var_5);
        		var_183 = wp::load(var_joint_q, var_182);
        		var_184 = wp::quat_from_axis_angle(var_181, var_183);
        		var_185 = wp::vec3();
        		var_186 = wp::mul(var_184, var_180);
        		var_187 = wp::transform(var_185, var_186);
        		var_188 = wp::add(var_23, var_2);
        		var_189 = wp::load(var_joint_qd, var_188);
        		var_190 = wp::mul(var_177, var_189);
        		var_191 = wp::add(var_23, var_5);
        		var_192 = wp::load(var_joint_qd, var_191);
        		var_193 = wp::mul(var_181, var_192);
        		var_194 = wp::add(var_190, var_193);
        		var_195 = wp::vec3();
        		var_196 = wp::spatial_vector(var_194, var_195);
        		wp::copy(var_197, var_187);
        		wp::copy(var_198, var_196);
        	}
        	var_199 = wp::select(var_171, var_166, var_197);
        	var_200 = wp::select(var_171, var_167, var_198);
        	var_201 = wp::select(var_171, var_168, var_187);
        	var_202 = wp::select(var_171, var_169, var_196);
        	var_203 = wp::select(var_171, var_125, var_172);
        	var_204 = wp::select(var_171, var_129, var_174);
        	var_205 = wp::select(var_171, var_131, var_176);
        	var_206 = wp::select(var_171, var_134, var_177);
        	var_207 = wp::select(var_171, var_137, var_180);
        	var_208 = wp::select(var_171, var_138, var_181);
        	var_209 = wp::select(var_171, var_141, var_184);
        	var_210 = wp::mul(var_16, var_20);
        	var_211 = wp::mul(var_210, var_199);
        	var_212 = wp::spatial_top(var_200);
        	var_213 = wp::transform_vector(var_210, var_212);
        	var_214 = wp::spatial_bottom(var_200);
        	var_215 = wp::transform_vector(var_210, var_214);
        	var_216 = wp::load(var_body_com, var_9);
        	var_217 = wp::cross(var_213, var_216);
        	var_218 = wp::add(var_215, var_217);
        	var_219 = wp::spatial_vector(var_213, var_218);
        	var_220 = wp::add(var_17, var_219);
        	wp::store(var_body_q, var_9, var_211);
        	wp::store(var_body_qd, var_9, var_220);
        	goto for_start_1;
        for_end_1:;

}


extern "C" __global__ void eval_articulation_fk_cuda_kernel_backward(launch_bounds_t dim,
	array_t<int32> var_articulation_start,
	array_t<int32> var_articulation_mask,
	array_t<float32> var_joint_q,
	array_t<float32> var_joint_qd,
	array_t<int32> var_joint_q_start,
	array_t<int32> var_joint_qd_start,
	array_t<int32> var_joint_type,
	array_t<int32> var_joint_parent,
	array_t<transform> var_joint_X_p,
	array_t<transform> var_joint_X_c,
	array_t<vec3> var_joint_axis,
	array_t<vec3> var_body_com,
	array_t<transform> var_body_q,
	array_t<spatial_vector> var_body_qd,
	array_t<int32> adj_articulation_start,
	array_t<int32> adj_articulation_mask,
	array_t<float32> adj_joint_q,
	array_t<float32> adj_joint_qd,
	array_t<int32> adj_joint_q_start,
	array_t<int32> adj_joint_qd_start,
	array_t<int32> adj_joint_type,
	array_t<int32> adj_joint_parent,
	array_t<transform> adj_joint_X_p,
	array_t<transform> adj_joint_X_c,
	array_t<vec3> adj_joint_axis,
	array_t<vec3> adj_body_com,
	array_t<transform> adj_body_q,
	array_t<spatial_vector> adj_body_qd)
{
    int _idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (_idx >= dim.size) 
        return;

    set_launch_bounds(dim);

        //---------
    // primal vars
    int32 var_0;
    int32 var_1;
    const int32 var_2 = 0;
    bool var_3;
    int32 var_4;
    const int32 var_5 = 1;
    int32 var_6;
    int32 var_7;
    range_t var_8;
    int32 var_9;
    int32 var_10;
    transform var_11;
    spatial_vector var_12;
    bool var_13;
    transform var_14;
    spatial_vector var_15;
    transform var_16;
    spatial_vector var_17;
    int32 var_18;
    vec3 var_19;
    transform var_20;
    transform var_21;
    int32 var_22;
    int32 var_23;
    const int32 var_24 = 0;
    bool var_25;
    float32 var_26;
    float32 var_27;
    vec3 var_28;
    quat var_29;
    transform var_30;
    vec3 var_31;
    vec3 var_32;
    spatial_vector var_33;
    const int32 var_34 = 1;
    bool var_35;
    float32 var_36;
    float32 var_37;
    vec3 var_38;
    quat var_39;
    transform var_40;
    vec3 var_41;
    vec3 var_42;
    spatial_vector var_43;
    float32 var_44;
    float32 var_45;
    transform var_46;
    spatial_vector var_47;
    const int32 var_48 = 2;
    bool var_49;
    int32 var_50;
    float32 var_51;
    int32 var_52;
    float32 var_53;
    const int32 var_54 = 2;
    int32 var_55;
    float32 var_56;
    const int32 var_57 = 3;
    int32 var_58;
    float32 var_59;
    quat var_60;
    int32 var_61;
    float32 var_62;
    int32 var_63;
    float32 var_64;
    int32 var_65;
    float32 var_66;
    vec3 var_67;
    vec3 var_68;
    transform var_69;
    vec3 var_70;
    spatial_vector var_71;
    transform var_72;
    spatial_vector var_73;
    const int32 var_74 = 3;
    bool var_75;
    transform var_76;
    vec3 var_77;
    vec3 var_78;
    spatial_vector var_79;
    transform var_80;
    spatial_vector var_81;
    const int32 var_82 = 4;
    bool var_83;
    int32 var_84;
    float32 var_85;
    int32 var_86;
    float32 var_87;
    int32 var_88;
    float32 var_89;
    vec3 var_90;
    int32 var_91;
    float32 var_92;
    const int32 var_93 = 4;
    int32 var_94;
    float32 var_95;
    const int32 var_96 = 5;
    int32 var_97;
    float32 var_98;
    const int32 var_99 = 6;
    int32 var_100;
    float32 var_101;
    quat var_102;
    transform var_103;
    int32 var_104;
    float32 var_105;
    int32 var_106;
    float32 var_107;
    int32 var_108;
    float32 var_109;
    vec3 var_110;
    int32 var_111;
    float32 var_112;
    int32 var_113;
    float32 var_114;
    int32 var_115;
    float32 var_116;
    vec3 var_117;
    spatial_vector var_118;
    transform var_119;
    spatial_vector var_120;
    transform var_121;
    spatial_vector var_122;
    const int32 var_123 = 5;
    bool var_124;
    quat var_125;
    const float32 var_126 = 1.0;
    const float32 var_127 = 0.0;
    vec3 var_128;
    vec3 var_129;
    vec3 var_130;
    vec3 var_131;
    vec3 var_132;
    vec3 var_133;
    vec3 var_134;
    int32 var_135;
    float32 var_136;
    quat var_137;
    vec3 var_138;
    int32 var_139;
    float32 var_140;
    quat var_141;
    quat var_142;
    vec3 var_143;
    int32 var_144;
    float32 var_145;
    quat var_146;
    vec3 var_147;
    quat var_148;
    quat var_149;
    transform var_150;
    int32 var_151;
    float32 var_152;
    vec3 var_153;
    int32 var_154;
    float32 var_155;
    vec3 var_156;
    vec3 var_157;
    int32 var_158;
    float32 var_159;
    vec3 var_160;
    vec3 var_161;
    vec3 var_162;
    spatial_vector var_163;
    transform var_164;
    spatial_vector var_165;
    transform var_166;
    spatial_vector var_167;
    transform var_168;
    spatial_vector var_169;
    const int32 var_170 = 6;
    bool var_171;
    quat var_172;
    vec3 var_173;
    vec3 var_174;
    vec3 var_175;
    vec3 var_176;
    vec3 var_177;
    int32 var_178;
    float32 var_179;
    quat var_180;
    vec3 var_181;
    int32 var_182;
    float32 var_183;
    quat var_184;
    vec3 var_185;
    quat var_186;
    transform var_187;
    int32 var_188;
    float32 var_189;
    vec3 var_190;
    int32 var_191;
    float32 var_192;
    vec3 var_193;
    vec3 var_194;
    vec3 var_195;
    spatial_vector var_196;
    transform var_197;
    spatial_vector var_198;
    transform var_199;
    spatial_vector var_200;
    transform var_201;
    spatial_vector var_202;
    quat var_203;
    vec3 var_204;
    vec3 var_205;
    vec3 var_206;
    quat var_207;
    vec3 var_208;
    quat var_209;
    transform var_210;
    transform var_211;
    vec3 var_212;
    vec3 var_213;
    vec3 var_214;
    vec3 var_215;
    vec3 var_216;
    vec3 var_217;
    vec3 var_218;
    spatial_vector var_219;
    spatial_vector var_220;
    //---------
    // dual vars
    int32 adj_0 = 0;
    int32 adj_1 = 0;
    int32 adj_2 = 0;
    bool adj_3 = 0;
    int32 adj_4 = 0;
    int32 adj_5 = 0;
    int32 adj_6 = 0;
    int32 adj_7 = 0;
    range_t adj_8 = 0;
    int32 adj_9 = 0;
    int32 adj_10 = 0;
    transform adj_11 = 0;
    spatial_vector adj_12 = 0;
    bool adj_13 = 0;
    transform adj_14 = 0;
    spatial_vector adj_15 = 0;
    transform adj_16 = 0;
    spatial_vector adj_17 = 0;
    int32 adj_18 = 0;
    vec3 adj_19 = 0;
    transform adj_20 = 0;
    transform adj_21 = 0;
    int32 adj_22 = 0;
    int32 adj_23 = 0;
    int32 adj_24 = 0;
    bool adj_25 = 0;
    float32 adj_26 = 0;
    float32 adj_27 = 0;
    vec3 adj_28 = 0;
    quat adj_29 = 0;
    transform adj_30 = 0;
    vec3 adj_31 = 0;
    vec3 adj_32 = 0;
    spatial_vector adj_33 = 0;
    int32 adj_34 = 0;
    bool adj_35 = 0;
    float32 adj_36 = 0;
    float32 adj_37 = 0;
    vec3 adj_38 = 0;
    quat adj_39 = 0;
    transform adj_40 = 0;
    vec3 adj_41 = 0;
    vec3 adj_42 = 0;
    spatial_vector adj_43 = 0;
    float32 adj_44 = 0;
    float32 adj_45 = 0;
    transform adj_46 = 0;
    spatial_vector adj_47 = 0;
    int32 adj_48 = 0;
    bool adj_49 = 0;
    int32 adj_50 = 0;
    float32 adj_51 = 0;
    int32 adj_52 = 0;
    float32 adj_53 = 0;
    int32 adj_54 = 0;
    int32 adj_55 = 0;
    float32 adj_56 = 0;
    int32 adj_57 = 0;
    int32 adj_58 = 0;
    float32 adj_59 = 0;
    quat adj_60 = 0;
    int32 adj_61 = 0;
    float32 adj_62 = 0;
    int32 adj_63 = 0;
    float32 adj_64 = 0;
    int32 adj_65 = 0;
    float32 adj_66 = 0;
    vec3 adj_67 = 0;
    vec3 adj_68 = 0;
    transform adj_69 = 0;
    vec3 adj_70 = 0;
    spatial_vector adj_71 = 0;
    transform adj_72 = 0;
    spatial_vector adj_73 = 0;
    int32 adj_74 = 0;
    bool adj_75 = 0;
    transform adj_76 = 0;
    vec3 adj_77 = 0;
    vec3 adj_78 = 0;
    spatial_vector adj_79 = 0;
    transform adj_80 = 0;
    spatial_vector adj_81 = 0;
    int32 adj_82 = 0;
    bool adj_83 = 0;
    int32 adj_84 = 0;
    float32 adj_85 = 0;
    int32 adj_86 = 0;
    float32 adj_87 = 0;
    int32 adj_88 = 0;
    float32 adj_89 = 0;
    vec3 adj_90 = 0;
    int32 adj_91 = 0;
    float32 adj_92 = 0;
    int32 adj_93 = 0;
    int32 adj_94 = 0;
    float32 adj_95 = 0;
    int32 adj_96 = 0;
    int32 adj_97 = 0;
    float32 adj_98 = 0;
    int32 adj_99 = 0;
    int32 adj_100 = 0;
    float32 adj_101 = 0;
    quat adj_102 = 0;
    transform adj_103 = 0;
    int32 adj_104 = 0;
    float32 adj_105 = 0;
    int32 adj_106 = 0;
    float32 adj_107 = 0;
    int32 adj_108 = 0;
    float32 adj_109 = 0;
    vec3 adj_110 = 0;
    int32 adj_111 = 0;
    float32 adj_112 = 0;
    int32 adj_113 = 0;
    float32 adj_114 = 0;
    int32 adj_115 = 0;
    float32 adj_116 = 0;
    vec3 adj_117 = 0;
    spatial_vector adj_118 = 0;
    transform adj_119 = 0;
    spatial_vector adj_120 = 0;
    transform adj_121 = 0;
    spatial_vector adj_122 = 0;
    int32 adj_123 = 0;
    bool adj_124 = 0;
    quat adj_125 = 0;
    float32 adj_126 = 0;
    float32 adj_127 = 0;
    vec3 adj_128 = 0;
    vec3 adj_129 = 0;
    vec3 adj_130 = 0;
    vec3 adj_131 = 0;
    vec3 adj_132 = 0;
    vec3 adj_133 = 0;
    vec3 adj_134 = 0;
    int32 adj_135 = 0;
    float32 adj_136 = 0;
    quat adj_137 = 0;
    vec3 adj_138 = 0;
    int32 adj_139 = 0;
    float32 adj_140 = 0;
    quat adj_141 = 0;
    quat adj_142 = 0;
    vec3 adj_143 = 0;
    int32 adj_144 = 0;
    float32 adj_145 = 0;
    quat adj_146 = 0;
    vec3 adj_147 = 0;
    quat adj_148 = 0;
    quat adj_149 = 0;
    transform adj_150 = 0;
    int32 adj_151 = 0;
    float32 adj_152 = 0;
    vec3 adj_153 = 0;
    int32 adj_154 = 0;
    float32 adj_155 = 0;
    vec3 adj_156 = 0;
    vec3 adj_157 = 0;
    int32 adj_158 = 0;
    float32 adj_159 = 0;
    vec3 adj_160 = 0;
    vec3 adj_161 = 0;
    vec3 adj_162 = 0;
    spatial_vector adj_163 = 0;
    transform adj_164 = 0;
    spatial_vector adj_165 = 0;
    transform adj_166 = 0;
    spatial_vector adj_167 = 0;
    transform adj_168 = 0;
    spatial_vector adj_169 = 0;
    int32 adj_170 = 0;
    bool adj_171 = 0;
    quat adj_172 = 0;
    vec3 adj_173 = 0;
    vec3 adj_174 = 0;
    vec3 adj_175 = 0;
    vec3 adj_176 = 0;
    vec3 adj_177 = 0;
    int32 adj_178 = 0;
    float32 adj_179 = 0;
    quat adj_180 = 0;
    vec3 adj_181 = 0;
    int32 adj_182 = 0;
    float32 adj_183 = 0;
    quat adj_184 = 0;
    vec3 adj_185 = 0;
    quat adj_186 = 0;
    transform adj_187 = 0;
    int32 adj_188 = 0;
    float32 adj_189 = 0;
    vec3 adj_190 = 0;
    int32 adj_191 = 0;
    float32 adj_192 = 0;
    vec3 adj_193 = 0;
    vec3 adj_194 = 0;
    vec3 adj_195 = 0;
    spatial_vector adj_196 = 0;
    transform adj_197 = 0;
    spatial_vector adj_198 = 0;
    transform adj_199 = 0;
    spatial_vector adj_200 = 0;
    transform adj_201 = 0;
    spatial_vector adj_202 = 0;
    quat adj_203 = 0;
    vec3 adj_204 = 0;
    vec3 adj_205 = 0;
    vec3 adj_206 = 0;
    quat adj_207 = 0;
    vec3 adj_208 = 0;
    quat adj_209 = 0;
    transform adj_210 = 0;
    transform adj_211 = 0;
    vec3 adj_212 = 0;
    vec3 adj_213 = 0;
    vec3 adj_214 = 0;
    vec3 adj_215 = 0;
    vec3 adj_216 = 0;
    vec3 adj_217 = 0;
    vec3 adj_218 = 0;
    spatial_vector adj_219 = 0;
    spatial_vector adj_220 = 0;
        //---------
        // forward
        var_0 = wp::tid();
        if (var_articulation_mask) {
        	var_1 = wp::load(var_articulation_mask, var_0);
        	var_3 = (var_1 == var_2);
        	if (var_3) {
        		goto label0;
        	}
        }
        var_4 = wp::load(var_articulation_start, var_0);
        var_6 = wp::add(var_0, var_5);
        var_7 = wp::load(var_articulation_start, var_6);
        var_8 = wp::range(var_4, var_7);
        //---------
        // reverse
        var_8 = wp::iter_reverse(var_8);
        for_start_1:;
        	if (iter_cmp(var_8) == 0) goto for_end_1;
        	var_9 = wp::iter_next(var_8);
        	adj_10 = 0;
        	adj_11 = 0;
        	adj_12 = 0;
        	adj_13 = 0;
        	adj_14 = 0;
        	adj_15 = 0;
        	adj_16 = 0;
        	adj_17 = 0;
        	adj_18 = 0;
        	adj_19 = 0;
        	adj_20 = 0;
        	adj_21 = 0;
        	adj_22 = 0;
        	adj_23 = 0;
        	adj_24 = 0;
        	adj_25 = 0;
        	adj_26 = 0;
        	adj_27 = 0;
        	adj_28 = 0;
        	adj_29 = 0;
        	adj_30 = 0;
        	adj_31 = 0;
        	adj_32 = 0;
        	adj_33 = 0;
        	adj_34 = 0;
        	adj_35 = 0;
        	adj_36 = 0;
        	adj_37 = 0;
        	adj_38 = 0;
        	adj_39 = 0;
        	adj_40 = 0;
        	adj_41 = 0;
        	adj_42 = 0;
        	adj_43 = 0;
        	adj_44 = 0;
        	adj_45 = 0;
        	adj_46 = 0;
        	adj_47 = 0;
        	adj_48 = 0;
        	adj_49 = 0;
        	adj_50 = 0;
        	adj_51 = 0;
        	adj_52 = 0;
        	adj_53 = 0;
        	adj_54 = 0;
        	adj_55 = 0;
        	adj_56 = 0;
        	adj_57 = 0;
        	adj_58 = 0;
        	adj_59 = 0;
        	adj_60 = 0;
        	adj_61 = 0;
        	adj_62 = 0;
        	adj_63 = 0;
        	adj_64 = 0;
        	adj_65 = 0;
        	adj_66 = 0;
        	adj_67 = 0;
        	adj_68 = 0;
        	adj_69 = 0;
        	adj_70 = 0;
        	adj_71 = 0;
        	adj_72 = 0;
        	adj_73 = 0;
        	adj_74 = 0;
        	adj_75 = 0;
        	adj_76 = 0;
        	adj_77 = 0;
        	adj_78 = 0;
        	adj_79 = 0;
        	adj_80 = 0;
        	adj_81 = 0;
        	adj_82 = 0;
        	adj_83 = 0;
        	adj_84 = 0;
        	adj_85 = 0;
        	adj_86 = 0;
        	adj_87 = 0;
        	adj_88 = 0;
        	adj_89 = 0;
        	adj_90 = 0;
        	adj_91 = 0;
        	adj_92 = 0;
        	adj_93 = 0;
        	adj_94 = 0;
        	adj_95 = 0;
        	adj_96 = 0;
        	adj_97 = 0;
        	adj_98 = 0;
        	adj_99 = 0;
        	adj_100 = 0;
        	adj_101 = 0;
        	adj_102 = 0;
        	adj_103 = 0;
        	adj_104 = 0;
        	adj_105 = 0;
        	adj_106 = 0;
        	adj_107 = 0;
        	adj_108 = 0;
        	adj_109 = 0;
        	adj_110 = 0;
        	adj_111 = 0;
        	adj_112 = 0;
        	adj_113 = 0;
        	adj_114 = 0;
        	adj_115 = 0;
        	adj_116 = 0;
        	adj_117 = 0;
        	adj_118 = 0;
        	adj_119 = 0;
        	adj_120 = 0;
        	adj_121 = 0;
        	adj_122 = 0;
        	adj_123 = 0;
        	adj_124 = 0;
        	adj_125 = 0;
        	adj_126 = 0;
        	adj_127 = 0;
        	adj_128 = 0;
        	adj_129 = 0;
        	adj_130 = 0;
        	adj_131 = 0;
        	adj_132 = 0;
        	adj_133 = 0;
        	adj_134 = 0;
        	adj_135 = 0;
        	adj_136 = 0;
        	adj_137 = 0;
        	adj_138 = 0;
        	adj_139 = 0;
        	adj_140 = 0;
        	adj_141 = 0;
        	adj_142 = 0;
        	adj_143 = 0;
        	adj_144 = 0;
        	adj_145 = 0;
        	adj_146 = 0;
        	adj_147 = 0;
        	adj_148 = 0;
        	adj_149 = 0;
        	adj_150 = 0;
        	adj_151 = 0;
        	adj_152 = 0;
        	adj_153 = 0;
        	adj_154 = 0;
        	adj_155 = 0;
        	adj_156 = 0;
        	adj_157 = 0;
        	adj_158 = 0;
        	adj_159 = 0;
        	adj_160 = 0;
        	adj_161 = 0;
        	adj_162 = 0;
        	adj_163 = 0;
        	adj_164 = 0;
        	adj_165 = 0;
        	adj_166 = 0;
        	adj_167 = 0;
        	adj_168 = 0;
        	adj_169 = 0;
        	adj_170 = 0;
        	adj_171 = 0;
        	adj_172 = 0;
        	adj_173 = 0;
        	adj_174 = 0;
        	adj_175 = 0;
        	adj_176 = 0;
        	adj_177 = 0;
        	adj_178 = 0;
        	adj_179 = 0;
        	adj_180 = 0;
        	adj_181 = 0;
        	adj_182 = 0;
        	adj_183 = 0;
        	adj_184 = 0;
        	adj_185 = 0;
        	adj_186 = 0;
        	adj_187 = 0;
        	adj_188 = 0;
        	adj_189 = 0;
        	adj_190 = 0;
        	adj_191 = 0;
        	adj_192 = 0;
        	adj_193 = 0;
        	adj_194 = 0;
        	adj_195 = 0;
        	adj_196 = 0;
        	adj_197 = 0;
        	adj_198 = 0;
        	adj_199 = 0;
        	adj_200 = 0;
        	adj_201 = 0;
        	adj_202 = 0;
        	adj_203 = 0;
        	adj_204 = 0;
        	adj_205 = 0;
        	adj_206 = 0;
        	adj_207 = 0;
        	adj_208 = 0;
        	adj_209 = 0;
        	adj_210 = 0;
        	adj_211 = 0;
        	adj_212 = 0;
        	adj_213 = 0;
        	adj_214 = 0;
        	adj_215 = 0;
        	adj_216 = 0;
        	adj_217 = 0;
        	adj_218 = 0;
        	adj_219 = 0;
        	adj_220 = 0;
        	var_10 = wp::load(var_joint_parent, var_9);
        	var_11 = wp::transform_identity();
        	var_12 = wp::spatial_vector();
        	var_13 = (var_10 >= var_2);
        	if (var_13) {
        		var_14 = wp::load(var_body_q, var_10);
        		var_15 = wp::load(var_body_qd, var_10);
        	}
        	var_16 = wp::select(var_13, var_11, var_14);
        	var_17 = wp::select(var_13, var_12, var_15);
        	var_18 = wp::load(var_joint_type, var_9);
        	var_19 = wp::load(var_joint_axis, var_9);
        	var_20 = wp::load(var_joint_X_p, var_9);
        	var_21 = wp::load(var_joint_X_c, var_9);
        	var_22 = wp::load(var_joint_q_start, var_9);
        	var_23 = wp::load(var_joint_qd_start, var_9);
        	var_25 = (var_18 == var_24);
        	if (var_25) {
        		var_26 = wp::load(var_joint_q, var_22);
        		var_27 = wp::load(var_joint_qd, var_23);
        		var_28 = wp::mul(var_19, var_26);
        		var_29 = wp::quat_identity();
        		var_30 = wp::transform(var_28, var_29);
        		var_31 = wp::vec3();
        		var_32 = wp::mul(var_19, var_27);
        		var_33 = wp::spatial_vector(var_31, var_32);
        	}
        	var_35 = (var_18 == var_34);
        	if (var_35) {
        		var_36 = wp::load(var_joint_q, var_22);
        		var_37 = wp::load(var_joint_qd, var_23);
        		var_38 = wp::vec3();
        		var_39 = wp::quat_from_axis_angle(var_19, var_36);
        		var_40 = wp::transform(var_38, var_39);
        		var_41 = wp::mul(var_19, var_37);
        		var_42 = wp::vec3();
        		var_43 = wp::spatial_vector(var_41, var_42);
        	}
        	var_44 = wp::select(var_35, var_26, var_36);
        	var_45 = wp::select(var_35, var_27, var_37);
        	var_46 = wp::select(var_35, var_30, var_40);
        	var_47 = wp::select(var_35, var_33, var_43);
        	var_49 = (var_18 == var_48);
        	if (var_49) {
        		var_50 = wp::add(var_22, var_2);
        		var_51 = wp::load(var_joint_q, var_50);
        		var_52 = wp::add(var_22, var_5);
        		var_53 = wp::load(var_joint_q, var_52);
        		var_55 = wp::add(var_22, var_54);
        		var_56 = wp::load(var_joint_q, var_55);
        		var_58 = wp::add(var_22, var_57);
        		var_59 = wp::load(var_joint_q, var_58);
        		var_60 = wp::quat(var_51, var_53, var_56, var_59);
        		var_61 = wp::add(var_23, var_2);
        		var_62 = wp::load(var_joint_qd, var_61);
        		var_63 = wp::add(var_23, var_5);
        		var_64 = wp::load(var_joint_qd, var_63);
        		var_65 = wp::add(var_23, var_54);
        		var_66 = wp::load(var_joint_qd, var_65);
        		var_67 = wp::vec3(var_62, var_64, var_66);
        		var_68 = wp::vec3();
        		var_69 = wp::transform(var_68, var_60);
        		var_70 = wp::vec3();
        		var_71 = wp::spatial_vector(var_67, var_70);
        	}
        	var_72 = wp::select(var_49, var_46, var_69);
        	var_73 = wp::select(var_49, var_47, var_71);
        	var_75 = (var_18 == var_74);
        	if (var_75) {
        		var_76 = wp::transform_identity();
        		var_77 = wp::vec3();
        		var_78 = wp::vec3();
        		var_79 = wp::spatial_vector(var_77, var_78);
        	}
        	var_80 = wp::select(var_75, var_72, var_76);
        	var_81 = wp::select(var_75, var_73, var_79);
        	var_83 = (var_18 == var_82);
        	if (var_83) {
        		var_84 = wp::add(var_22, var_2);
        		var_85 = wp::load(var_joint_q, var_84);
        		var_86 = wp::add(var_22, var_5);
        		var_87 = wp::load(var_joint_q, var_86);
        		var_88 = wp::add(var_22, var_54);
        		var_89 = wp::load(var_joint_q, var_88);
        		var_90 = wp::vec3(var_85, var_87, var_89);
        		var_91 = wp::add(var_22, var_57);
        		var_92 = wp::load(var_joint_q, var_91);
        		var_94 = wp::add(var_22, var_93);
        		var_95 = wp::load(var_joint_q, var_94);
        		var_97 = wp::add(var_22, var_96);
        		var_98 = wp::load(var_joint_q, var_97);
        		var_100 = wp::add(var_22, var_99);
        		var_101 = wp::load(var_joint_q, var_100);
        		var_102 = wp::quat(var_92, var_95, var_98, var_101);
        		var_103 = wp::transform(var_90, var_102);
        		var_104 = wp::add(var_23, var_2);
        		var_105 = wp::load(var_joint_qd, var_104);
        		var_106 = wp::add(var_23, var_5);
        		var_107 = wp::load(var_joint_qd, var_106);
        		var_108 = wp::add(var_23, var_54);
        		var_109 = wp::load(var_joint_qd, var_108);
        		var_110 = wp::vec3(var_105, var_107, var_109);
        		var_111 = wp::add(var_23, var_57);
        		var_112 = wp::load(var_joint_qd, var_111);
        		var_113 = wp::add(var_23, var_93);
        		var_114 = wp::load(var_joint_qd, var_113);
        		var_115 = wp::add(var_23, var_96);
        		var_116 = wp::load(var_joint_qd, var_115);
        		var_117 = wp::vec3(var_112, var_114, var_116);
        		var_118 = wp::spatial_vector(var_110, var_117);
        		wp::copy(var_119, var_103);
        		wp::copy(var_120, var_118);
        	}
        	var_121 = wp::select(var_83, var_80, var_119);
        	var_122 = wp::select(var_83, var_81, var_120);
        	var_124 = (var_18 == var_123);
        	if (var_124) {
        		var_125 = wp::transform_get_rotation(var_21);
        		var_128 = wp::vec3(var_126, var_127, var_127);
        		var_129 = wp::quat_rotate(var_125, var_128);
        		var_130 = wp::vec3(var_127, var_126, var_127);
        		var_131 = wp::quat_rotate(var_125, var_130);
        		var_132 = wp::vec3(var_127, var_127, var_126);
        		var_133 = wp::quat_rotate(var_125, var_132);
        		wp::copy(var_134, var_129);
        		var_135 = wp::add(var_22, var_2);
        		var_136 = wp::load(var_joint_q, var_135);
        		var_137 = wp::quat_from_axis_angle(var_134, var_136);
        		var_138 = wp::quat_rotate(var_137, var_131);
        		var_139 = wp::add(var_22, var_5);
        		var_140 = wp::load(var_joint_q, var_139);
        		var_141 = wp::quat_from_axis_angle(var_138, var_140);
        		var_142 = wp::mul(var_141, var_137);
        		var_143 = wp::quat_rotate(var_142, var_133);
        		var_144 = wp::add(var_22, var_54);
        		var_145 = wp::load(var_joint_q, var_144);
        		var_146 = wp::quat_from_axis_angle(var_143, var_145);
        		var_147 = wp::vec3();
        		var_148 = wp::mul(var_146, var_141);
        		var_149 = wp::mul(var_148, var_137);
        		var_150 = wp::transform(var_147, var_149);
        		var_151 = wp::add(var_23, var_2);
        		var_152 = wp::load(var_joint_qd, var_151);
        		var_153 = wp::mul(var_134, var_152);
        		var_154 = wp::add(var_23, var_5);
        		var_155 = wp::load(var_joint_qd, var_154);
        		var_156 = wp::mul(var_138, var_155);
        		var_157 = wp::add(var_153, var_156);
        		var_158 = wp::add(var_23, var_54);
        		var_159 = wp::load(var_joint_qd, var_158);
        		var_160 = wp::mul(var_143, var_159);
        		var_161 = wp::add(var_157, var_160);
        		var_162 = wp::vec3();
        		var_163 = wp::spatial_vector(var_161, var_162);
        		wp::copy(var_164, var_150);
        		wp::copy(var_165, var_163);
        	}
        	var_166 = wp::select(var_124, var_121, var_164);
        	var_167 = wp::select(var_124, var_122, var_165);
        	var_168 = wp::select(var_124, var_103, var_150);
        	var_169 = wp::select(var_124, var_118, var_163);
        	var_171 = (var_18 == var_170);
        	if (var_171) {
        		var_172 = wp::transform_get_rotation(var_21);
        		var_173 = wp::vec3(var_126, var_127, var_127);
        		var_174 = wp::quat_rotate(var_172, var_173);
        		var_175 = wp::vec3(var_127, var_126, var_127);
        		var_176 = wp::quat_rotate(var_172, var_175);
        		wp::copy(var_177, var_174);
        		var_178 = wp::add(var_22, var_2);
        		var_179 = wp::load(var_joint_q, var_178);
        		var_180 = wp::quat_from_axis_angle(var_177, var_179);
        		var_181 = wp::quat_rotate(var_180, var_176);
        		var_182 = wp::add(var_22, var_5);
        		var_183 = wp::load(var_joint_q, var_182);
        		var_184 = wp::quat_from_axis_angle(var_181, var_183);
        		var_185 = wp::vec3();
        		var_186 = wp::mul(var_184, var_180);
        		var_187 = wp::transform(var_185, var_186);
        		var_188 = wp::add(var_23, var_2);
        		var_189 = wp::load(var_joint_qd, var_188);
        		var_190 = wp::mul(var_177, var_189);
        		var_191 = wp::add(var_23, var_5);
        		var_192 = wp::load(var_joint_qd, var_191);
        		var_193 = wp::mul(var_181, var_192);
        		var_194 = wp::add(var_190, var_193);
        		var_195 = wp::vec3();
        		var_196 = wp::spatial_vector(var_194, var_195);
        		wp::copy(var_197, var_187);
        		wp::copy(var_198, var_196);
        	}
        	var_199 = wp::select(var_171, var_166, var_197);
        	var_200 = wp::select(var_171, var_167, var_198);
        	var_201 = wp::select(var_171, var_168, var_187);
        	var_202 = wp::select(var_171, var_169, var_196);
        	var_203 = wp::select(var_171, var_125, var_172);
        	var_204 = wp::select(var_171, var_129, var_174);
        	var_205 = wp::select(var_171, var_131, var_176);
        	var_206 = wp::select(var_171, var_134, var_177);
        	var_207 = wp::select(var_171, var_137, var_180);
        	var_208 = wp::select(var_171, var_138, var_181);
        	var_209 = wp::select(var_171, var_141, var_184);
        	var_210 = wp::mul(var_16, var_20);
        	var_211 = wp::mul(var_210, var_199);
        	var_212 = wp::spatial_top(var_200);
        	var_213 = wp::transform_vector(var_210, var_212);
        	var_214 = wp::spatial_bottom(var_200);
        	var_215 = wp::transform_vector(var_210, var_214);
        	var_216 = wp::load(var_body_com, var_9);
        	var_217 = wp::cross(var_213, var_216);
        	var_218 = wp::add(var_215, var_217);
        	var_219 = wp::spatial_vector(var_213, var_218);
        	var_220 = wp::add(var_17, var_219);
        	//wp::store(var_body_q, var_9, var_211);
        	//wp::store(var_body_qd, var_9, var_220);
        	wp::adj_store(var_body_qd, var_9, var_220, adj_body_qd, adj_9, adj_220);
        	wp::adj_store(var_body_q, var_9, var_211, adj_body_q, adj_9, adj_211);
        	wp::adj_add(var_17, var_219, adj_17, adj_219, adj_220);
        	wp::adj_spatial_vector(var_213, var_218, adj_213, adj_218, adj_219);
        	wp::adj_add(var_215, var_217, adj_215, adj_217, adj_218);
        	wp::adj_cross(var_213, var_216, adj_213, adj_216, adj_217);
        	wp::adj_load(var_body_com, var_9, adj_body_com, adj_9, adj_216);
        	wp::adj_transform_vector(var_210, var_214, adj_210, adj_214, adj_215);
        	wp::adj_spatial_bottom(var_200, adj_200, adj_214);
        	wp::adj_transform_vector(var_210, var_212, adj_210, adj_212, adj_213);
        	wp::adj_spatial_top(var_200, adj_200, adj_212);
        	wp::adj_mul(var_210, var_199, adj_210, adj_199, adj_211);
        	wp::adj_mul(var_16, var_20, adj_16, adj_20, adj_210);
        	wp::adj_select(var_171, var_141, var_184, adj_171, adj_141, adj_184, adj_209);
        	wp::adj_select(var_171, var_138, var_181, adj_171, adj_138, adj_181, adj_208);
        	wp::adj_select(var_171, var_137, var_180, adj_171, adj_137, adj_180, adj_207);
        	wp::adj_select(var_171, var_134, var_177, adj_171, adj_134, adj_177, adj_206);
        	wp::adj_select(var_171, var_131, var_176, adj_171, adj_131, adj_176, adj_205);
        	wp::adj_select(var_171, var_129, var_174, adj_171, adj_129, adj_174, adj_204);
        	wp::adj_select(var_171, var_125, var_172, adj_171, adj_125, adj_172, adj_203);
        	wp::adj_select(var_171, var_169, var_196, adj_171, adj_169, adj_196, adj_202);
        	wp::adj_select(var_171, var_168, var_187, adj_171, adj_168, adj_187, adj_201);
        	wp::adj_select(var_171, var_167, var_198, adj_171, adj_167, adj_198, adj_200);
        	wp::adj_select(var_171, var_166, var_197, adj_171, adj_166, adj_197, adj_199);
        	if (var_171) {
        		wp::adj_copy(var_198, var_196, adj_198, adj_196);
        		wp::adj_copy(var_197, var_187, adj_197, adj_187);
        		wp::adj_spatial_vector(var_194, var_195, adj_194, adj_195, adj_196);
        		wp::adj_add(var_190, var_193, adj_190, adj_193, adj_194);
        		wp::adj_mul(var_181, var_192, adj_181, adj_192, adj_193);
        		wp::adj_load(var_joint_qd, var_191, adj_joint_qd, adj_191, adj_192);
        		wp::adj_add(var_23, var_5, adj_23, adj_5, adj_191);
        		wp::adj_mul(var_177, var_189, adj_177, adj_189, adj_190);
        		wp::adj_load(var_joint_qd, var_188, adj_joint_qd, adj_188, adj_189);
        		wp::adj_add(var_23, var_2, adj_23, adj_2, adj_188);
        		wp::adj_transform(var_185, var_186, adj_185, adj_186, adj_187);
        		wp::adj_mul(var_184, var_180, adj_184, adj_180, adj_186);
        		wp::adj_quat_from_axis_angle(var_181, var_183, adj_181, adj_183, adj_184);
        		wp::adj_load(var_joint_q, var_182, adj_joint_q, adj_182, adj_183);
        		wp::adj_add(var_22, var_5, adj_22, adj_5, adj_182);
        		wp::adj_quat_rotate(var_180, var_176, adj_180, adj_176, adj_181);
        		wp::adj_quat_from_axis_angle(var_177, var_179, adj_177, adj_179, adj_180);
        		wp::adj_load(var_joint_q, var_178, adj_joint_q, adj_178, adj_179);
        		wp::adj_add(var_22, var_2, adj_22, adj_2, adj_178);
        		wp::adj_copy(var_177, var_174, adj_177, adj_174);
        		wp::adj_quat_rotate(var_172, var_175, adj_172, adj_175, adj_176);
        		wp::adj_vec3(var_127, var_126, var_127, adj_127, adj_126, adj_127, adj_175);
        		wp::adj_quat_rotate(var_172, var_173, adj_172, adj_173, adj_174);
        		wp::adj_vec3(var_126, var_127, var_127, adj_126, adj_127, adj_127, adj_173);
        		wp::adj_transform_get_rotation(var_21, adj_21, adj_172);
        	}
        	wp::adj_select(var_124, var_118, var_163, adj_124, adj_118, adj_163, adj_169);
        	wp::adj_select(var_124, var_103, var_150, adj_124, adj_103, adj_150, adj_168);
        	wp::adj_select(var_124, var_122, var_165, adj_124, adj_122, adj_165, adj_167);
        	wp::adj_select(var_124, var_121, var_164, adj_124, adj_121, adj_164, adj_166);
        	if (var_124) {
        		wp::adj_copy(var_165, var_163, adj_165, adj_163);
        		wp::adj_copy(var_164, var_150, adj_164, adj_150);
        		wp::adj_spatial_vector(var_161, var_162, adj_161, adj_162, adj_163);
        		wp::adj_add(var_157, var_160, adj_157, adj_160, adj_161);
        		wp::adj_mul(var_143, var_159, adj_143, adj_159, adj_160);
        		wp::adj_load(var_joint_qd, var_158, adj_joint_qd, adj_158, adj_159);
        		wp::adj_add(var_23, var_54, adj_23, adj_54, adj_158);
        		wp::adj_add(var_153, var_156, adj_153, adj_156, adj_157);
        		wp::adj_mul(var_138, var_155, adj_138, adj_155, adj_156);
        		wp::adj_load(var_joint_qd, var_154, adj_joint_qd, adj_154, adj_155);
        		wp::adj_add(var_23, var_5, adj_23, adj_5, adj_154);
        		wp::adj_mul(var_134, var_152, adj_134, adj_152, adj_153);
        		wp::adj_load(var_joint_qd, var_151, adj_joint_qd, adj_151, adj_152);
        		wp::adj_add(var_23, var_2, adj_23, adj_2, adj_151);
        		wp::adj_transform(var_147, var_149, adj_147, adj_149, adj_150);
        		wp::adj_mul(var_148, var_137, adj_148, adj_137, adj_149);
        		wp::adj_mul(var_146, var_141, adj_146, adj_141, adj_148);
        		wp::adj_quat_from_axis_angle(var_143, var_145, adj_143, adj_145, adj_146);
        		wp::adj_load(var_joint_q, var_144, adj_joint_q, adj_144, adj_145);
        		wp::adj_add(var_22, var_54, adj_22, adj_54, adj_144);
        		wp::adj_quat_rotate(var_142, var_133, adj_142, adj_133, adj_143);
        		wp::adj_mul(var_141, var_137, adj_141, adj_137, adj_142);
        		wp::adj_quat_from_axis_angle(var_138, var_140, adj_138, adj_140, adj_141);
        		wp::adj_load(var_joint_q, var_139, adj_joint_q, adj_139, adj_140);
        		wp::adj_add(var_22, var_5, adj_22, adj_5, adj_139);
        		wp::adj_quat_rotate(var_137, var_131, adj_137, adj_131, adj_138);
        		wp::adj_quat_from_axis_angle(var_134, var_136, adj_134, adj_136, adj_137);
        		wp::adj_load(var_joint_q, var_135, adj_joint_q, adj_135, adj_136);
        		wp::adj_add(var_22, var_2, adj_22, adj_2, adj_135);
        		wp::adj_copy(var_134, var_129, adj_134, adj_129);
        		wp::adj_quat_rotate(var_125, var_132, adj_125, adj_132, adj_133);
        		wp::adj_vec3(var_127, var_127, var_126, adj_127, adj_127, adj_126, adj_132);
        		wp::adj_quat_rotate(var_125, var_130, adj_125, adj_130, adj_131);
        		wp::adj_vec3(var_127, var_126, var_127, adj_127, adj_126, adj_127, adj_130);
        		wp::adj_quat_rotate(var_125, var_128, adj_125, adj_128, adj_129);
        		wp::adj_vec3(var_126, var_127, var_127, adj_126, adj_127, adj_127, adj_128);
        		wp::adj_transform_get_rotation(var_21, adj_21, adj_125);
        	}
        	wp::adj_select(var_83, var_81, var_120, adj_83, adj_81, adj_120, adj_122);
        	wp::adj_select(var_83, var_80, var_119, adj_83, adj_80, adj_119, adj_121);
        	if (var_83) {
        		wp::adj_copy(var_120, var_118, adj_120, adj_118);
        		wp::adj_copy(var_119, var_103, adj_119, adj_103);
        		wp::adj_spatial_vector(var_110, var_117, adj_110, adj_117, adj_118);
        		wp::adj_vec3(var_112, var_114, var_116, adj_112, adj_114, adj_116, adj_117);
        		wp::adj_load(var_joint_qd, var_115, adj_joint_qd, adj_115, adj_116);
        		wp::adj_add(var_23, var_96, adj_23, adj_96, adj_115);
        		wp::adj_load(var_joint_qd, var_113, adj_joint_qd, adj_113, adj_114);
        		wp::adj_add(var_23, var_93, adj_23, adj_93, adj_113);
        		wp::adj_load(var_joint_qd, var_111, adj_joint_qd, adj_111, adj_112);
        		wp::adj_add(var_23, var_57, adj_23, adj_57, adj_111);
        		wp::adj_vec3(var_105, var_107, var_109, adj_105, adj_107, adj_109, adj_110);
        		wp::adj_load(var_joint_qd, var_108, adj_joint_qd, adj_108, adj_109);
        		wp::adj_add(var_23, var_54, adj_23, adj_54, adj_108);
        		wp::adj_load(var_joint_qd, var_106, adj_joint_qd, adj_106, adj_107);
        		wp::adj_add(var_23, var_5, adj_23, adj_5, adj_106);
        		wp::adj_load(var_joint_qd, var_104, adj_joint_qd, adj_104, adj_105);
        		wp::adj_add(var_23, var_2, adj_23, adj_2, adj_104);
        		wp::adj_transform(var_90, var_102, adj_90, adj_102, adj_103);
        		wp::adj_quat(var_92, var_95, var_98, var_101, adj_92, adj_95, adj_98, adj_101, adj_102);
        		wp::adj_load(var_joint_q, var_100, adj_joint_q, adj_100, adj_101);
        		wp::adj_add(var_22, var_99, adj_22, adj_99, adj_100);
        		wp::adj_load(var_joint_q, var_97, adj_joint_q, adj_97, adj_98);
        		wp::adj_add(var_22, var_96, adj_22, adj_96, adj_97);
        		wp::adj_load(var_joint_q, var_94, adj_joint_q, adj_94, adj_95);
        		wp::adj_add(var_22, var_93, adj_22, adj_93, adj_94);
        		wp::adj_load(var_joint_q, var_91, adj_joint_q, adj_91, adj_92);
        		wp::adj_add(var_22, var_57, adj_22, adj_57, adj_91);
        		wp::adj_vec3(var_85, var_87, var_89, adj_85, adj_87, adj_89, adj_90);
        		wp::adj_load(var_joint_q, var_88, adj_joint_q, adj_88, adj_89);
        		wp::adj_add(var_22, var_54, adj_22, adj_54, adj_88);
        		wp::adj_load(var_joint_q, var_86, adj_joint_q, adj_86, adj_87);
        		wp::adj_add(var_22, var_5, adj_22, adj_5, adj_86);
        		wp::adj_load(var_joint_q, var_84, adj_joint_q, adj_84, adj_85);
        		wp::adj_add(var_22, var_2, adj_22, adj_2, adj_84);
        	}
        	wp::adj_select(var_75, var_73, var_79, adj_75, adj_73, adj_79, adj_81);
        	wp::adj_select(var_75, var_72, var_76, adj_75, adj_72, adj_76, adj_80);
        	if (var_75) {
        		wp::adj_spatial_vector(var_77, var_78, adj_77, adj_78, adj_79);
        	}
        	wp::adj_select(var_49, var_47, var_71, adj_49, adj_47, adj_71, adj_73);
        	wp::adj_select(var_49, var_46, var_69, adj_49, adj_46, adj_69, adj_72);
        	if (var_49) {
        		wp::adj_spatial_vector(var_67, var_70, adj_67, adj_70, adj_71);
        		wp::adj_transform(var_68, var_60, adj_68, adj_60, adj_69);
        		wp::adj_vec3(var_62, var_64, var_66, adj_62, adj_64, adj_66, adj_67);
        		wp::adj_load(var_joint_qd, var_65, adj_joint_qd, adj_65, adj_66);
        		wp::adj_add(var_23, var_54, adj_23, adj_54, adj_65);
        		wp::adj_load(var_joint_qd, var_63, adj_joint_qd, adj_63, adj_64);
        		wp::adj_add(var_23, var_5, adj_23, adj_5, adj_63);
        		wp::adj_load(var_joint_qd, var_61, adj_joint_qd, adj_61, adj_62);
        		wp::adj_add(var_23, var_2, adj_23, adj_2, adj_61);
        		wp::adj_quat(var_51, var_53, var_56, var_59, adj_51, adj_53, adj_56, adj_59, adj_60);
        		wp::adj_load(var_joint_q, var_58, adj_joint_q, adj_58, adj_59);
        		wp::adj_add(var_22, var_57, adj_22, adj_57, adj_58);
        		wp::adj_load(var_joint_q, var_55, adj_joint_q, adj_55, adj_56);
        		wp::adj_add(var_22, var_54, adj_22, adj_54, adj_55);
        		wp::adj_load(var_joint_q, var_52, adj_joint_q, adj_52, adj_53);
        		wp::adj_add(var_22, var_5, adj_22, adj_5, adj_52);
        		wp::adj_load(var_joint_q, var_50, adj_joint_q, adj_50, adj_51);
        		wp::adj_add(var_22, var_2, adj_22, adj_2, adj_50);
        	}
        	wp::adj_select(var_35, var_33, var_43, adj_35, adj_33, adj_43, adj_47);
        	wp::adj_select(var_35, var_30, var_40, adj_35, adj_30, adj_40, adj_46);
        	wp::adj_select(var_35, var_27, var_37, adj_35, adj_27, adj_37, adj_45);
        	wp::adj_select(var_35, var_26, var_36, adj_35, adj_26, adj_36, adj_44);
        	if (var_35) {
        		wp::adj_spatial_vector(var_41, var_42, adj_41, adj_42, adj_43);
        		wp::adj_mul(var_19, var_37, adj_19, adj_37, adj_41);
        		wp::adj_transform(var_38, var_39, adj_38, adj_39, adj_40);
        		wp::adj_quat_from_axis_angle(var_19, var_36, adj_19, adj_36, adj_39);
        		wp::adj_load(var_joint_qd, var_23, adj_joint_qd, adj_23, adj_37);
        		wp::adj_load(var_joint_q, var_22, adj_joint_q, adj_22, adj_36);
        	}
        	if (var_25) {
        		wp::adj_spatial_vector(var_31, var_32, adj_31, adj_32, adj_33);
        		wp::adj_mul(var_19, var_27, adj_19, adj_27, adj_32);
        		wp::adj_transform(var_28, var_29, adj_28, adj_29, adj_30);
        		wp::adj_mul(var_19, var_26, adj_19, adj_26, adj_28);
        		wp::adj_load(var_joint_qd, var_23, adj_joint_qd, adj_23, adj_27);
        		wp::adj_load(var_joint_q, var_22, adj_joint_q, adj_22, adj_26);
        	}
        	wp::adj_load(var_joint_qd_start, var_9, adj_joint_qd_start, adj_9, adj_23);
        	wp::adj_load(var_joint_q_start, var_9, adj_joint_q_start, adj_9, adj_22);
        	wp::adj_load(var_joint_X_c, var_9, adj_joint_X_c, adj_9, adj_21);
        	wp::adj_load(var_joint_X_p, var_9, adj_joint_X_p, adj_9, adj_20);
        	wp::adj_load(var_joint_axis, var_9, adj_joint_axis, adj_9, adj_19);
        	wp::adj_load(var_joint_type, var_9, adj_joint_type, adj_9, adj_18);
        	wp::adj_select(var_13, var_12, var_15, adj_13, adj_12, adj_15, adj_17);
        	wp::adj_select(var_13, var_11, var_14, adj_13, adj_11, adj_14, adj_16);
        	if (var_13) {
        		wp::adj_load(var_body_qd, var_10, adj_body_qd, adj_10, adj_15);
        		wp::adj_load(var_body_q, var_10, adj_body_q, adj_10, adj_14);
        	}
        	wp::adj_load(var_joint_parent, var_9, adj_joint_parent, adj_9, adj_10);
        	goto for_start_1;
        for_end_1:;
        wp::adj_range(var_4, var_7, adj_4, adj_7, adj_8);
        wp::adj_load(var_articulation_start, var_6, adj_articulation_start, adj_6, adj_7);
        wp::adj_add(var_0, var_5, adj_0, adj_5, adj_6);
        wp::adj_load(var_articulation_start, var_0, adj_articulation_start, adj_0, adj_4);
        if (var_articulation_mask) {
        	if (var_3) {
        		label0:;
        	}
        	wp::adj_load(var_articulation_mask, var_0, adj_articulation_mask, adj_0, adj_1);
        }
        return;

}



extern "C" {

// Python entry points
WP_API void eval_articulation_fk_cuda_forward(void* stream, launch_bounds_t dim,
	array_t<int32> var_articulation_start,
	array_t<int32> var_articulation_mask,
	array_t<float32> var_joint_q,
	array_t<float32> var_joint_qd,
	array_t<int32> var_joint_q_start,
	array_t<int32> var_joint_qd_start,
	array_t<int32> var_joint_type,
	array_t<int32> var_joint_parent,
	array_t<transform> var_joint_X_p,
	array_t<transform> var_joint_X_c,
	array_t<vec3> var_joint_axis,
	array_t<vec3> var_body_com,
	array_t<transform> var_body_q,
	array_t<spatial_vector> var_body_qd)
{
    eval_articulation_fk_cuda_kernel_forward<<<(dim.size + 256 - 1) / 256, 256, 0, (cudaStream_t)stream>>>(dim,
			var_articulation_start,
			var_articulation_mask,
			var_joint_q,
			var_joint_qd,
			var_joint_q_start,
			var_joint_qd_start,
			var_joint_type,
			var_joint_parent,
			var_joint_X_p,
			var_joint_X_c,
			var_joint_axis,
			var_body_com,
			var_body_q,
			var_body_qd);
}

WP_API void eval_articulation_fk_cuda_backward(void* stream, launch_bounds_t dim,
	array_t<int32> var_articulation_start,
	array_t<int32> var_articulation_mask,
	array_t<float32> var_joint_q,
	array_t<float32> var_joint_qd,
	array_t<int32> var_joint_q_start,
	array_t<int32> var_joint_qd_start,
	array_t<int32> var_joint_type,
	array_t<int32> var_joint_parent,
	array_t<transform> var_joint_X_p,
	array_t<transform> var_joint_X_c,
	array_t<vec3> var_joint_axis,
	array_t<vec3> var_body_com,
	array_t<transform> var_body_q,
	array_t<spatial_vector> var_body_qd,
	array_t<int32> adj_articulation_start,
	array_t<int32> adj_articulation_mask,
	array_t<float32> adj_joint_q,
	array_t<float32> adj_joint_qd,
	array_t<int32> adj_joint_q_start,
	array_t<int32> adj_joint_qd_start,
	array_t<int32> adj_joint_type,
	array_t<int32> adj_joint_parent,
	array_t<transform> adj_joint_X_p,
	array_t<transform> adj_joint_X_c,
	array_t<vec3> adj_joint_axis,
	array_t<vec3> adj_body_com,
	array_t<transform> adj_body_q,
	array_t<spatial_vector> adj_body_qd)
{
    eval_articulation_fk_cuda_kernel_backward<<<(dim.size + 256 - 1) / 256, 256, 0, (cudaStream_t)stream>>>(dim,
			var_articulation_start,
			var_articulation_mask,
			var_joint_q,
			var_joint_qd,
			var_joint_q_start,
			var_joint_qd_start,
			var_joint_type,
			var_joint_parent,
			var_joint_X_p,
			var_joint_X_c,
			var_joint_axis,
			var_body_com,
			var_body_q,
			var_body_qd,
			adj_articulation_start,
			adj_articulation_mask,
			adj_joint_q,
			adj_joint_qd,
			adj_joint_q_start,
			adj_joint_qd_start,
			adj_joint_type,
			adj_joint_parent,
			adj_joint_X_p,
			adj_joint_X_c,
			adj_joint_axis,
			adj_body_com,
			adj_body_q,
			adj_body_qd);
}

} // extern C



extern "C" __global__ void eval_articulation_ik_cuda_kernel_forward(launch_bounds_t dim,
	array_t<transform> var_body_q,
	array_t<spatial_vector> var_body_qd,
	array_t<vec3> var_body_com,
	array_t<int32> var_joint_type,
	array_t<int32> var_joint_parent,
	array_t<transform> var_joint_X_p,
	array_t<transform> var_joint_X_c,
	array_t<vec3> var_joint_axis,
	array_t<int32> var_joint_q_start,
	array_t<int32> var_joint_qd_start,
	array_t<float32> var_joint_q,
	array_t<float32> var_joint_qd)
{
    int _idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (_idx >= dim.size) 
        return;

    set_launch_bounds(dim);

        //---------
    // primal vars
    int32 var_0;
    int32 var_1;
    int32 var_2;
    transform var_3;
    transform var_4;
    transform var_5;
    vec3 var_6;
    vec3 var_7;
    vec3 var_8;
    const int32 var_9 = 0;
    bool var_10;
    transform var_11;
    transform var_12;
    vec3 var_13;
    transform var_14;
    vec3 var_15;
    vec3 var_16;
    vec3 var_17;
    spatial_vector var_18;
    vec3 var_19;
    vec3 var_20;
    vec3 var_21;
    vec3 var_22;
    transform var_23;
    vec3 var_24;
    vec3 var_25;
    transform var_26;
    vec3 var_27;
    transform var_28;
    vec3 var_29;
    vec3 var_30;
    vec3 var_31;
    spatial_vector var_32;
    vec3 var_33;
    vec3 var_34;
    vec3 var_35;
    vec3 var_36;
    int32 var_37;
    vec3 var_38;
    vec3 var_39;
    vec3 var_40;
    quat var_41;
    quat var_42;
    vec3 var_43;
    vec3 var_44;
    vec3 var_45;
    int32 var_46;
    int32 var_47;
    const int32 var_48 = 0;
    bool var_49;
    vec3 var_50;
    float32 var_51;
    float32 var_52;
    const int32 var_53 = 1;
    bool var_54;
    vec3 var_55;
    vec3 var_56;
    quat var_57;
    quat var_58;
    quat var_59;
    const int32 var_60 = 3;
    float32 var_61;
    float32 var_62;
    const float32 var_63 = 2.0;
    float32 var_64;
    float32 var_65;
    const int32 var_66 = 1;
    float32 var_67;
    const int32 var_68 = 2;
    float32 var_69;
    vec3 var_70;
    float32 var_71;
    float32 var_72;
    float32 var_73;
    float32 var_74;
    vec3 var_75;
    float32 var_76;
    float32 var_77;
    const int32 var_78 = 2;
    bool var_79;
    quat var_80;
    quat var_81;
    float32 var_82;
    int32 var_83;
    float32 var_84;
    int32 var_85;
    float32 var_86;
    int32 var_87;
    float32 var_88;
    int32 var_89;
    float32 var_90;
    int32 var_91;
    float32 var_92;
    int32 var_93;
    float32 var_94;
    int32 var_95;
    quat var_96;
    const int32 var_97 = 3;
    bool var_98;
    const int32 var_99 = 4;
    bool var_100;
    quat var_101;
    quat var_102;
    float32 var_103;
    int32 var_104;
    float32 var_105;
    int32 var_106;
    float32 var_107;
    int32 var_108;
    float32 var_109;
    int32 var_110;
    float32 var_111;
    const int32 var_112 = 4;
    int32 var_113;
    float32 var_114;
    const int32 var_115 = 5;
    int32 var_116;
    float32 var_117;
    const int32 var_118 = 6;
    int32 var_119;
    float32 var_120;
    int32 var_121;
    float32 var_122;
    int32 var_123;
    float32 var_124;
    int32 var_125;
    float32 var_126;
    int32 var_127;
    float32 var_128;
    int32 var_129;
    float32 var_130;
    int32 var_131;
    quat var_132;
    const int32 var_133 = 5;
    bool var_134;
    quat var_135;
    quat var_136;
    quat var_137;
    quat var_138;
    quat var_139;
    quat var_140;
    vec3 var_141;
    const float32 var_142 = 1.0;
    const float32 var_143 = 0.0;
    vec3 var_144;
    float32 var_145;
    quat var_146;
    vec3 var_147;
    vec3 var_148;
    float32 var_149;
    quat var_150;
    quat var_151;
    vec3 var_152;
    vec3 var_153;
    float32 var_154;
    quat var_155;
    quat var_156;
    float32 var_157;
    int32 var_158;
    float32 var_159;
    int32 var_160;
    float32 var_161;
    int32 var_162;
    vec3 var_163;
    float32 var_164;
    int32 var_165;
    vec3 var_166;
    float32 var_167;
    int32 var_168;
    vec3 var_169;
    float32 var_170;
    int32 var_171;
    quat var_172;
    const int32 var_173 = 6;
    bool var_174;
    quat var_175;
    quat var_176;
    quat var_177;
    quat var_178;
    quat var_179;
    quat var_180;
    vec3 var_181;
    vec3 var_182;
    float32 var_183;
    quat var_184;
    vec3 var_185;
    vec3 var_186;
    float32 var_187;
    quat var_188;
    quat var_189;
    float32 var_190;
    int32 var_191;
    float32 var_192;
    int32 var_193;
    vec3 var_194;
    float32 var_195;
    int32 var_196;
    vec3 var_197;
    float32 var_198;
    int32 var_199;
    quat var_200;
    quat var_201;
    vec3 var_202;
    vec3 var_203;
    quat var_204;
    vec3 var_205;
    quat var_206;
    quat var_207;
    //---------
    // forward
        var_0 = wp::tid();
        wp::copy(var_1, var_0);
        var_2 = wp::load(var_joint_parent, var_0);
        var_3 = wp::load(var_joint_X_p, var_0);
        var_4 = wp::load(var_joint_X_c, var_0);
        wp::copy(var_5, var_3);
        var_6 = wp::vec3();
        var_7 = wp::vec3();
        var_8 = wp::vec3();
        var_10 = (var_2 >= var_9);
        if (var_10) {
        	var_11 = wp::load(var_body_q, var_2);
        	var_12 = wp::mul(var_11, var_5);
        	var_13 = wp::transform_get_translation(var_12);
        	var_14 = wp::load(var_body_q, var_2);
        	var_15 = wp::load(var_body_com, var_2);
        	var_16 = wp::transform_point(var_14, var_15);
        	var_17 = wp::sub(var_13, var_16);
        	var_18 = wp::load(var_body_qd, var_2);
        	var_19 = wp::spatial_top(var_18);
        	var_20 = wp::spatial_bottom(var_18);
        	var_21 = wp::cross(var_19, var_17);
        	var_22 = wp::add(var_20, var_21);
        }
        var_23 = wp::select(var_10, var_5, var_12);
        var_24 = wp::select(var_10, var_7, var_19);
        var_25 = wp::select(var_10, var_8, var_22);
        var_26 = wp::load(var_body_q, var_1);
        var_27 = wp::transform_get_translation(var_26);
        var_28 = wp::load(var_body_q, var_1);
        var_29 = wp::load(var_body_com, var_1);
        var_30 = wp::transform_point(var_28, var_29);
        var_31 = wp::sub(var_27, var_30);
        var_32 = wp::load(var_body_qd, var_1);
        var_33 = wp::spatial_top(var_32);
        var_34 = wp::spatial_bottom(var_32);
        var_35 = wp::cross(var_33, var_31);
        var_36 = wp::add(var_34, var_35);
        var_37 = wp::load(var_joint_type, var_0);
        var_38 = wp::load(var_joint_axis, var_0);
        var_39 = wp::transform_get_translation(var_23);
        var_40 = wp::transform_get_translation(var_26);
        var_41 = wp::transform_get_rotation(var_23);
        var_42 = wp::transform_get_rotation(var_26);
        var_43 = wp::sub(var_40, var_39);
        var_44 = wp::sub(var_36, var_25);
        var_45 = wp::sub(var_33, var_24);
        var_46 = wp::load(var_joint_q_start, var_0);
        var_47 = wp::load(var_joint_qd_start, var_0);
        var_49 = (var_37 == var_48);
        if (var_49) {
        	var_50 = wp::transform_vector(var_23, var_38);
        	var_51 = wp::dot(var_43, var_50);
        	var_52 = wp::dot(var_44, var_50);
        	wp::store(var_joint_q, var_46, var_51);
        	wp::store(var_joint_qd, var_47, var_52);
        	return;
        }
        var_54 = (var_37 == var_53);
        if (var_54) {
        	var_55 = wp::transform_vector(var_23, var_38);
        	var_56 = wp::transform_vector(var_26, var_38);
        	var_57 = wp::quat_inverse(var_41);
        	var_58 = wp::mul(var_57, var_42);
        	var_59 = quat_twist(var_38, var_58);
        	var_61 = wp::index(var_59, var_60);
        	var_62 = wp::acos(var_61);
        	var_64 = wp::mul(var_62, var_63);
        	var_65 = wp::index(var_59, var_9);
        	var_67 = wp::index(var_59, var_66);
        	var_69 = wp::index(var_59, var_68);
        	var_70 = wp::vec3(var_65, var_67, var_69);
        	var_71 = wp::dot(var_38, var_70);
        	var_72 = wp::sign(var_71);
        	var_73 = wp::mul(var_64, var_72);
        	var_74 = wp::dot(var_45, var_55);
        	wp::store(var_joint_q, var_46, var_73);
        	wp::store(var_joint_qd, var_47, var_74);
        	return;
        }
        var_75 = wp::select(var_54, var_50, var_55);
        var_76 = wp::select(var_54, var_51, var_73);
        var_77 = wp::select(var_54, var_52, var_74);
        var_79 = (var_37 == var_78);
        if (var_79) {
        	var_80 = wp::quat_inverse(var_41);
        	var_81 = wp::mul(var_80, var_42);
        	var_82 = wp::index(var_81, var_9);
        	var_83 = wp::add(var_46, var_9);
        	wp::store(var_joint_q, var_83, var_82);
        	var_84 = wp::index(var_81, var_66);
        	var_85 = wp::add(var_46, var_66);
        	wp::store(var_joint_q, var_85, var_84);
        	var_86 = wp::index(var_81, var_68);
        	var_87 = wp::add(var_46, var_68);
        	wp::store(var_joint_q, var_87, var_86);
        	var_88 = wp::index(var_81, var_60);
        	var_89 = wp::add(var_46, var_60);
        	wp::store(var_joint_q, var_89, var_88);
        	var_90 = wp::index(var_45, var_9);
        	var_91 = wp::add(var_47, var_9);
        	wp::store(var_joint_qd, var_91, var_90);
        	var_92 = wp::index(var_45, var_66);
        	var_93 = wp::add(var_47, var_66);
        	wp::store(var_joint_qd, var_93, var_92);
        	var_94 = wp::index(var_45, var_68);
        	var_95 = wp::add(var_47, var_68);
        	wp::store(var_joint_qd, var_95, var_94);
        	return;
        }
        var_96 = wp::select(var_79, var_58, var_81);
        var_98 = (var_37 == var_97);
        if (var_98) {
        	return;
        }
        var_100 = (var_37 == var_99);
        if (var_100) {
        	var_101 = wp::quat_inverse(var_41);
        	var_102 = wp::mul(var_101, var_42);
        	var_103 = wp::index(var_43, var_9);
        	var_104 = wp::add(var_46, var_9);
        	wp::store(var_joint_q, var_104, var_103);
        	var_105 = wp::index(var_43, var_66);
        	var_106 = wp::add(var_46, var_66);
        	wp::store(var_joint_q, var_106, var_105);
        	var_107 = wp::index(var_43, var_68);
        	var_108 = wp::add(var_46, var_68);
        	wp::store(var_joint_q, var_108, var_107);
        	var_109 = wp::index(var_102, var_9);
        	var_110 = wp::add(var_46, var_60);
        	wp::store(var_joint_q, var_110, var_109);
        	var_111 = wp::index(var_102, var_66);
        	var_113 = wp::add(var_46, var_112);
        	wp::store(var_joint_q, var_113, var_111);
        	var_114 = wp::index(var_102, var_68);
        	var_116 = wp::add(var_46, var_115);
        	wp::store(var_joint_q, var_116, var_114);
        	var_117 = wp::index(var_102, var_60);
        	var_119 = wp::add(var_46, var_118);
        	wp::store(var_joint_q, var_119, var_117);
        	var_120 = wp::index(var_45, var_9);
        	var_121 = wp::add(var_47, var_9);
        	wp::store(var_joint_qd, var_121, var_120);
        	var_122 = wp::index(var_45, var_66);
        	var_123 = wp::add(var_47, var_66);
        	wp::store(var_joint_qd, var_123, var_122);
        	var_124 = wp::index(var_45, var_68);
        	var_125 = wp::add(var_47, var_68);
        	wp::store(var_joint_qd, var_125, var_124);
        	var_126 = wp::index(var_44, var_9);
        	var_127 = wp::add(var_47, var_60);
        	wp::store(var_joint_qd, var_127, var_126);
        	var_128 = wp::index(var_44, var_66);
        	var_129 = wp::add(var_47, var_112);
        	wp::store(var_joint_qd, var_129, var_128);
        	var_130 = wp::index(var_44, var_68);
        	var_131 = wp::add(var_47, var_115);
        	wp::store(var_joint_qd, var_131, var_130);
        }
        var_132 = wp::select(var_100, var_96, var_102);
        var_134 = (var_37 == var_133);
        if (var_134) {
        	var_135 = wp::transform_get_rotation(var_4);
        	var_136 = wp::quat_inverse(var_135);
        	var_137 = wp::quat_inverse(var_41);
        	var_138 = wp::mul(var_136, var_137);
        	var_139 = wp::mul(var_138, var_42);
        	var_140 = wp::mul(var_139, var_135);
        	var_141 = quat_decompose(var_140);
        	var_144 = wp::vec3(var_142, var_143, var_143);
        	var_145 = wp::index(var_141, var_9);
        	var_146 = wp::quat_from_axis_angle(var_144, var_145);
        	var_147 = wp::vec3(var_143, var_142, var_143);
        	var_148 = wp::quat_rotate(var_146, var_147);
        	var_149 = wp::index(var_141, var_66);
        	var_150 = wp::quat_from_axis_angle(var_148, var_149);
        	var_151 = wp::mul(var_150, var_146);
        	var_152 = wp::vec3(var_143, var_143, var_142);
        	var_153 = wp::quat_rotate(var_151, var_152);
        	var_154 = wp::index(var_141, var_68);
        	var_155 = wp::quat_from_axis_angle(var_153, var_154);
        	var_156 = wp::mul(var_41, var_135);
        	var_157 = wp::index(var_141, var_9);
        	var_158 = wp::add(var_46, var_9);
        	wp::store(var_joint_q, var_158, var_157);
        	var_159 = wp::index(var_141, var_66);
        	var_160 = wp::add(var_46, var_66);
        	wp::store(var_joint_q, var_160, var_159);
        	var_161 = wp::index(var_141, var_68);
        	var_162 = wp::add(var_46, var_68);
        	wp::store(var_joint_q, var_162, var_161);
        	var_163 = wp::quat_rotate(var_156, var_144);
        	var_164 = wp::dot(var_163, var_45);
        	var_165 = wp::add(var_47, var_9);
        	wp::store(var_joint_qd, var_165, var_164);
        	var_166 = wp::quat_rotate(var_156, var_148);
        	var_167 = wp::dot(var_166, var_45);
        	var_168 = wp::add(var_47, var_66);
        	wp::store(var_joint_qd, var_168, var_167);
        	var_169 = wp::quat_rotate(var_156, var_153);
        	var_170 = wp::dot(var_169, var_45);
        	var_171 = wp::add(var_47, var_68);
        	wp::store(var_joint_qd, var_171, var_170);
        	return;
        }
        var_172 = wp::select(var_134, var_132, var_140);
        var_174 = (var_37 == var_173);
        if (var_174) {
        	var_175 = wp::transform_get_rotation(var_4);
        	var_176 = wp::quat_inverse(var_175);
        	var_177 = wp::quat_inverse(var_41);
        	var_178 = wp::mul(var_176, var_177);
        	var_179 = wp::mul(var_178, var_42);
        	var_180 = wp::mul(var_179, var_175);
        	var_181 = quat_decompose(var_180);
        	var_182 = wp::vec3(var_142, var_143, var_143);
        	var_183 = wp::index(var_181, var_9);
        	var_184 = wp::quat_from_axis_angle(var_182, var_183);
        	var_185 = wp::vec3(var_143, var_142, var_143);
        	var_186 = wp::quat_rotate(var_184, var_185);
        	var_187 = wp::index(var_181, var_66);
        	var_188 = wp::quat_from_axis_angle(var_186, var_187);
        	var_189 = wp::mul(var_41, var_175);
        	var_190 = wp::index(var_181, var_9);
        	var_191 = wp::add(var_46, var_9);
        	wp::store(var_joint_q, var_191, var_190);
        	var_192 = wp::index(var_181, var_66);
        	var_193 = wp::add(var_46, var_66);
        	wp::store(var_joint_q, var_193, var_192);
        	var_194 = wp::quat_rotate(var_189, var_182);
        	var_195 = wp::dot(var_194, var_45);
        	var_196 = wp::add(var_47, var_9);
        	wp::store(var_joint_qd, var_196, var_195);
        	var_197 = wp::quat_rotate(var_189, var_186);
        	var_198 = wp::dot(var_197, var_45);
        	var_199 = wp::add(var_47, var_66);
        	wp::store(var_joint_qd, var_199, var_198);
        	return;
        }
        var_200 = wp::select(var_174, var_172, var_180);
        var_201 = wp::select(var_174, var_135, var_175);
        var_202 = wp::select(var_174, var_141, var_181);
        var_203 = wp::select(var_174, var_144, var_182);
        var_204 = wp::select(var_174, var_146, var_184);
        var_205 = wp::select(var_174, var_148, var_186);
        var_206 = wp::select(var_174, var_150, var_188);
        var_207 = wp::select(var_174, var_156, var_189);

}


extern "C" __global__ void eval_articulation_ik_cuda_kernel_backward(launch_bounds_t dim,
	array_t<transform> var_body_q,
	array_t<spatial_vector> var_body_qd,
	array_t<vec3> var_body_com,
	array_t<int32> var_joint_type,
	array_t<int32> var_joint_parent,
	array_t<transform> var_joint_X_p,
	array_t<transform> var_joint_X_c,
	array_t<vec3> var_joint_axis,
	array_t<int32> var_joint_q_start,
	array_t<int32> var_joint_qd_start,
	array_t<float32> var_joint_q,
	array_t<float32> var_joint_qd,
	array_t<transform> adj_body_q,
	array_t<spatial_vector> adj_body_qd,
	array_t<vec3> adj_body_com,
	array_t<int32> adj_joint_type,
	array_t<int32> adj_joint_parent,
	array_t<transform> adj_joint_X_p,
	array_t<transform> adj_joint_X_c,
	array_t<vec3> adj_joint_axis,
	array_t<int32> adj_joint_q_start,
	array_t<int32> adj_joint_qd_start,
	array_t<float32> adj_joint_q,
	array_t<float32> adj_joint_qd)
{
    int _idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (_idx >= dim.size) 
        return;

    set_launch_bounds(dim);

        //---------
    // primal vars
    int32 var_0;
    int32 var_1;
    int32 var_2;
    transform var_3;
    transform var_4;
    transform var_5;
    vec3 var_6;
    vec3 var_7;
    vec3 var_8;
    const int32 var_9 = 0;
    bool var_10;
    transform var_11;
    transform var_12;
    vec3 var_13;
    transform var_14;
    vec3 var_15;
    vec3 var_16;
    vec3 var_17;
    spatial_vector var_18;
    vec3 var_19;
    vec3 var_20;
    vec3 var_21;
    vec3 var_22;
    transform var_23;
    vec3 var_24;
    vec3 var_25;
    transform var_26;
    vec3 var_27;
    transform var_28;
    vec3 var_29;
    vec3 var_30;
    vec3 var_31;
    spatial_vector var_32;
    vec3 var_33;
    vec3 var_34;
    vec3 var_35;
    vec3 var_36;
    int32 var_37;
    vec3 var_38;
    vec3 var_39;
    vec3 var_40;
    quat var_41;
    quat var_42;
    vec3 var_43;
    vec3 var_44;
    vec3 var_45;
    int32 var_46;
    int32 var_47;
    const int32 var_48 = 0;
    bool var_49;
    vec3 var_50;
    float32 var_51;
    float32 var_52;
    const int32 var_53 = 1;
    bool var_54;
    vec3 var_55;
    vec3 var_56;
    quat var_57;
    quat var_58;
    quat var_59;
    const int32 var_60 = 3;
    float32 var_61;
    float32 var_62;
    const float32 var_63 = 2.0;
    float32 var_64;
    float32 var_65;
    const int32 var_66 = 1;
    float32 var_67;
    const int32 var_68 = 2;
    float32 var_69;
    vec3 var_70;
    float32 var_71;
    float32 var_72;
    float32 var_73;
    float32 var_74;
    vec3 var_75;
    float32 var_76;
    float32 var_77;
    const int32 var_78 = 2;
    bool var_79;
    quat var_80;
    quat var_81;
    float32 var_82;
    int32 var_83;
    float32 var_84;
    int32 var_85;
    float32 var_86;
    int32 var_87;
    float32 var_88;
    int32 var_89;
    float32 var_90;
    int32 var_91;
    float32 var_92;
    int32 var_93;
    float32 var_94;
    int32 var_95;
    quat var_96;
    const int32 var_97 = 3;
    bool var_98;
    const int32 var_99 = 4;
    bool var_100;
    quat var_101;
    quat var_102;
    float32 var_103;
    int32 var_104;
    float32 var_105;
    int32 var_106;
    float32 var_107;
    int32 var_108;
    float32 var_109;
    int32 var_110;
    float32 var_111;
    const int32 var_112 = 4;
    int32 var_113;
    float32 var_114;
    const int32 var_115 = 5;
    int32 var_116;
    float32 var_117;
    const int32 var_118 = 6;
    int32 var_119;
    float32 var_120;
    int32 var_121;
    float32 var_122;
    int32 var_123;
    float32 var_124;
    int32 var_125;
    float32 var_126;
    int32 var_127;
    float32 var_128;
    int32 var_129;
    float32 var_130;
    int32 var_131;
    quat var_132;
    const int32 var_133 = 5;
    bool var_134;
    quat var_135;
    quat var_136;
    quat var_137;
    quat var_138;
    quat var_139;
    quat var_140;
    vec3 var_141;
    const float32 var_142 = 1.0;
    const float32 var_143 = 0.0;
    vec3 var_144;
    float32 var_145;
    quat var_146;
    vec3 var_147;
    vec3 var_148;
    float32 var_149;
    quat var_150;
    quat var_151;
    vec3 var_152;
    vec3 var_153;
    float32 var_154;
    quat var_155;
    quat var_156;
    float32 var_157;
    int32 var_158;
    float32 var_159;
    int32 var_160;
    float32 var_161;
    int32 var_162;
    vec3 var_163;
    float32 var_164;
    int32 var_165;
    vec3 var_166;
    float32 var_167;
    int32 var_168;
    vec3 var_169;
    float32 var_170;
    int32 var_171;
    quat var_172;
    const int32 var_173 = 6;
    bool var_174;
    quat var_175;
    quat var_176;
    quat var_177;
    quat var_178;
    quat var_179;
    quat var_180;
    vec3 var_181;
    vec3 var_182;
    float32 var_183;
    quat var_184;
    vec3 var_185;
    vec3 var_186;
    float32 var_187;
    quat var_188;
    quat var_189;
    float32 var_190;
    int32 var_191;
    float32 var_192;
    int32 var_193;
    vec3 var_194;
    float32 var_195;
    int32 var_196;
    vec3 var_197;
    float32 var_198;
    int32 var_199;
    quat var_200;
    quat var_201;
    vec3 var_202;
    vec3 var_203;
    quat var_204;
    vec3 var_205;
    quat var_206;
    quat var_207;
    //---------
    // dual vars
    int32 adj_0 = 0;
    int32 adj_1 = 0;
    int32 adj_2 = 0;
    transform adj_3 = 0;
    transform adj_4 = 0;
    transform adj_5 = 0;
    vec3 adj_6 = 0;
    vec3 adj_7 = 0;
    vec3 adj_8 = 0;
    int32 adj_9 = 0;
    bool adj_10 = 0;
    transform adj_11 = 0;
    transform adj_12 = 0;
    vec3 adj_13 = 0;
    transform adj_14 = 0;
    vec3 adj_15 = 0;
    vec3 adj_16 = 0;
    vec3 adj_17 = 0;
    spatial_vector adj_18 = 0;
    vec3 adj_19 = 0;
    vec3 adj_20 = 0;
    vec3 adj_21 = 0;
    vec3 adj_22 = 0;
    transform adj_23 = 0;
    vec3 adj_24 = 0;
    vec3 adj_25 = 0;
    transform adj_26 = 0;
    vec3 adj_27 = 0;
    transform adj_28 = 0;
    vec3 adj_29 = 0;
    vec3 adj_30 = 0;
    vec3 adj_31 = 0;
    spatial_vector adj_32 = 0;
    vec3 adj_33 = 0;
    vec3 adj_34 = 0;
    vec3 adj_35 = 0;
    vec3 adj_36 = 0;
    int32 adj_37 = 0;
    vec3 adj_38 = 0;
    vec3 adj_39 = 0;
    vec3 adj_40 = 0;
    quat adj_41 = 0;
    quat adj_42 = 0;
    vec3 adj_43 = 0;
    vec3 adj_44 = 0;
    vec3 adj_45 = 0;
    int32 adj_46 = 0;
    int32 adj_47 = 0;
    int32 adj_48 = 0;
    bool adj_49 = 0;
    vec3 adj_50 = 0;
    float32 adj_51 = 0;
    float32 adj_52 = 0;
    int32 adj_53 = 0;
    bool adj_54 = 0;
    vec3 adj_55 = 0;
    vec3 adj_56 = 0;
    quat adj_57 = 0;
    quat adj_58 = 0;
    quat adj_59 = 0;
    int32 adj_60 = 0;
    float32 adj_61 = 0;
    float32 adj_62 = 0;
    float32 adj_63 = 0;
    float32 adj_64 = 0;
    float32 adj_65 = 0;
    int32 adj_66 = 0;
    float32 adj_67 = 0;
    int32 adj_68 = 0;
    float32 adj_69 = 0;
    vec3 adj_70 = 0;
    float32 adj_71 = 0;
    float32 adj_72 = 0;
    float32 adj_73 = 0;
    float32 adj_74 = 0;
    vec3 adj_75 = 0;
    float32 adj_76 = 0;
    float32 adj_77 = 0;
    int32 adj_78 = 0;
    bool adj_79 = 0;
    quat adj_80 = 0;
    quat adj_81 = 0;
    float32 adj_82 = 0;
    int32 adj_83 = 0;
    float32 adj_84 = 0;
    int32 adj_85 = 0;
    float32 adj_86 = 0;
    int32 adj_87 = 0;
    float32 adj_88 = 0;
    int32 adj_89 = 0;
    float32 adj_90 = 0;
    int32 adj_91 = 0;
    float32 adj_92 = 0;
    int32 adj_93 = 0;
    float32 adj_94 = 0;
    int32 adj_95 = 0;
    quat adj_96 = 0;
    int32 adj_97 = 0;
    bool adj_98 = 0;
    int32 adj_99 = 0;
    bool adj_100 = 0;
    quat adj_101 = 0;
    quat adj_102 = 0;
    float32 adj_103 = 0;
    int32 adj_104 = 0;
    float32 adj_105 = 0;
    int32 adj_106 = 0;
    float32 adj_107 = 0;
    int32 adj_108 = 0;
    float32 adj_109 = 0;
    int32 adj_110 = 0;
    float32 adj_111 = 0;
    int32 adj_112 = 0;
    int32 adj_113 = 0;
    float32 adj_114 = 0;
    int32 adj_115 = 0;
    int32 adj_116 = 0;
    float32 adj_117 = 0;
    int32 adj_118 = 0;
    int32 adj_119 = 0;
    float32 adj_120 = 0;
    int32 adj_121 = 0;
    float32 adj_122 = 0;
    int32 adj_123 = 0;
    float32 adj_124 = 0;
    int32 adj_125 = 0;
    float32 adj_126 = 0;
    int32 adj_127 = 0;
    float32 adj_128 = 0;
    int32 adj_129 = 0;
    float32 adj_130 = 0;
    int32 adj_131 = 0;
    quat adj_132 = 0;
    int32 adj_133 = 0;
    bool adj_134 = 0;
    quat adj_135 = 0;
    quat adj_136 = 0;
    quat adj_137 = 0;
    quat adj_138 = 0;
    quat adj_139 = 0;
    quat adj_140 = 0;
    vec3 adj_141 = 0;
    float32 adj_142 = 0;
    float32 adj_143 = 0;
    vec3 adj_144 = 0;
    float32 adj_145 = 0;
    quat adj_146 = 0;
    vec3 adj_147 = 0;
    vec3 adj_148 = 0;
    float32 adj_149 = 0;
    quat adj_150 = 0;
    quat adj_151 = 0;
    vec3 adj_152 = 0;
    vec3 adj_153 = 0;
    float32 adj_154 = 0;
    quat adj_155 = 0;
    quat adj_156 = 0;
    float32 adj_157 = 0;
    int32 adj_158 = 0;
    float32 adj_159 = 0;
    int32 adj_160 = 0;
    float32 adj_161 = 0;
    int32 adj_162 = 0;
    vec3 adj_163 = 0;
    float32 adj_164 = 0;
    int32 adj_165 = 0;
    vec3 adj_166 = 0;
    float32 adj_167 = 0;
    int32 adj_168 = 0;
    vec3 adj_169 = 0;
    float32 adj_170 = 0;
    int32 adj_171 = 0;
    quat adj_172 = 0;
    int32 adj_173 = 0;
    bool adj_174 = 0;
    quat adj_175 = 0;
    quat adj_176 = 0;
    quat adj_177 = 0;
    quat adj_178 = 0;
    quat adj_179 = 0;
    quat adj_180 = 0;
    vec3 adj_181 = 0;
    vec3 adj_182 = 0;
    float32 adj_183 = 0;
    quat adj_184 = 0;
    vec3 adj_185 = 0;
    vec3 adj_186 = 0;
    float32 adj_187 = 0;
    quat adj_188 = 0;
    quat adj_189 = 0;
    float32 adj_190 = 0;
    int32 adj_191 = 0;
    float32 adj_192 = 0;
    int32 adj_193 = 0;
    vec3 adj_194 = 0;
    float32 adj_195 = 0;
    int32 adj_196 = 0;
    vec3 adj_197 = 0;
    float32 adj_198 = 0;
    int32 adj_199 = 0;
    quat adj_200 = 0;
    quat adj_201 = 0;
    vec3 adj_202 = 0;
    vec3 adj_203 = 0;
    quat adj_204 = 0;
    vec3 adj_205 = 0;
    quat adj_206 = 0;
    quat adj_207 = 0;
        //---------
        // forward
        var_0 = wp::tid();
        wp::copy(var_1, var_0);
        var_2 = wp::load(var_joint_parent, var_0);
        var_3 = wp::load(var_joint_X_p, var_0);
        var_4 = wp::load(var_joint_X_c, var_0);
        wp::copy(var_5, var_3);
        var_6 = wp::vec3();
        var_7 = wp::vec3();
        var_8 = wp::vec3();
        var_10 = (var_2 >= var_9);
        if (var_10) {
        	var_11 = wp::load(var_body_q, var_2);
        	var_12 = wp::mul(var_11, var_5);
        	var_13 = wp::transform_get_translation(var_12);
        	var_14 = wp::load(var_body_q, var_2);
        	var_15 = wp::load(var_body_com, var_2);
        	var_16 = wp::transform_point(var_14, var_15);
        	var_17 = wp::sub(var_13, var_16);
        	var_18 = wp::load(var_body_qd, var_2);
        	var_19 = wp::spatial_top(var_18);
        	var_20 = wp::spatial_bottom(var_18);
        	var_21 = wp::cross(var_19, var_17);
        	var_22 = wp::add(var_20, var_21);
        }
        var_23 = wp::select(var_10, var_5, var_12);
        var_24 = wp::select(var_10, var_7, var_19);
        var_25 = wp::select(var_10, var_8, var_22);
        var_26 = wp::load(var_body_q, var_1);
        var_27 = wp::transform_get_translation(var_26);
        var_28 = wp::load(var_body_q, var_1);
        var_29 = wp::load(var_body_com, var_1);
        var_30 = wp::transform_point(var_28, var_29);
        var_31 = wp::sub(var_27, var_30);
        var_32 = wp::load(var_body_qd, var_1);
        var_33 = wp::spatial_top(var_32);
        var_34 = wp::spatial_bottom(var_32);
        var_35 = wp::cross(var_33, var_31);
        var_36 = wp::add(var_34, var_35);
        var_37 = wp::load(var_joint_type, var_0);
        var_38 = wp::load(var_joint_axis, var_0);
        var_39 = wp::transform_get_translation(var_23);
        var_40 = wp::transform_get_translation(var_26);
        var_41 = wp::transform_get_rotation(var_23);
        var_42 = wp::transform_get_rotation(var_26);
        var_43 = wp::sub(var_40, var_39);
        var_44 = wp::sub(var_36, var_25);
        var_45 = wp::sub(var_33, var_24);
        var_46 = wp::load(var_joint_q_start, var_0);
        var_47 = wp::load(var_joint_qd_start, var_0);
        var_49 = (var_37 == var_48);
        if (var_49) {
        	var_50 = wp::transform_vector(var_23, var_38);
        	var_51 = wp::dot(var_43, var_50);
        	var_52 = wp::dot(var_44, var_50);
        	//wp::store(var_joint_q, var_46, var_51);
        	//wp::store(var_joint_qd, var_47, var_52);
        	goto label0;
        }
        var_54 = (var_37 == var_53);
        if (var_54) {
        	var_55 = wp::transform_vector(var_23, var_38);
        	var_56 = wp::transform_vector(var_26, var_38);
        	var_57 = wp::quat_inverse(var_41);
        	var_58 = wp::mul(var_57, var_42);
        	var_59 = quat_twist(var_38, var_58);
        	var_61 = wp::index(var_59, var_60);
        	var_62 = wp::acos(var_61);
        	var_64 = wp::mul(var_62, var_63);
        	var_65 = wp::index(var_59, var_9);
        	var_67 = wp::index(var_59, var_66);
        	var_69 = wp::index(var_59, var_68);
        	var_70 = wp::vec3(var_65, var_67, var_69);
        	var_71 = wp::dot(var_38, var_70);
        	var_72 = wp::sign(var_71);
        	var_73 = wp::mul(var_64, var_72);
        	var_74 = wp::dot(var_45, var_55);
        	//wp::store(var_joint_q, var_46, var_73);
        	//wp::store(var_joint_qd, var_47, var_74);
        	goto label1;
        }
        var_75 = wp::select(var_54, var_50, var_55);
        var_76 = wp::select(var_54, var_51, var_73);
        var_77 = wp::select(var_54, var_52, var_74);
        var_79 = (var_37 == var_78);
        if (var_79) {
        	var_80 = wp::quat_inverse(var_41);
        	var_81 = wp::mul(var_80, var_42);
        	var_82 = wp::index(var_81, var_9);
        	var_83 = wp::add(var_46, var_9);
        	//wp::store(var_joint_q, var_83, var_82);
        	var_84 = wp::index(var_81, var_66);
        	var_85 = wp::add(var_46, var_66);
        	//wp::store(var_joint_q, var_85, var_84);
        	var_86 = wp::index(var_81, var_68);
        	var_87 = wp::add(var_46, var_68);
        	//wp::store(var_joint_q, var_87, var_86);
        	var_88 = wp::index(var_81, var_60);
        	var_89 = wp::add(var_46, var_60);
        	//wp::store(var_joint_q, var_89, var_88);
        	var_90 = wp::index(var_45, var_9);
        	var_91 = wp::add(var_47, var_9);
        	//wp::store(var_joint_qd, var_91, var_90);
        	var_92 = wp::index(var_45, var_66);
        	var_93 = wp::add(var_47, var_66);
        	//wp::store(var_joint_qd, var_93, var_92);
        	var_94 = wp::index(var_45, var_68);
        	var_95 = wp::add(var_47, var_68);
        	//wp::store(var_joint_qd, var_95, var_94);
        	goto label2;
        }
        var_96 = wp::select(var_79, var_58, var_81);
        var_98 = (var_37 == var_97);
        if (var_98) {
        	goto label3;
        }
        var_100 = (var_37 == var_99);
        if (var_100) {
        	var_101 = wp::quat_inverse(var_41);
        	var_102 = wp::mul(var_101, var_42);
        	var_103 = wp::index(var_43, var_9);
        	var_104 = wp::add(var_46, var_9);
        	//wp::store(var_joint_q, var_104, var_103);
        	var_105 = wp::index(var_43, var_66);
        	var_106 = wp::add(var_46, var_66);
        	//wp::store(var_joint_q, var_106, var_105);
        	var_107 = wp::index(var_43, var_68);
        	var_108 = wp::add(var_46, var_68);
        	//wp::store(var_joint_q, var_108, var_107);
        	var_109 = wp::index(var_102, var_9);
        	var_110 = wp::add(var_46, var_60);
        	//wp::store(var_joint_q, var_110, var_109);
        	var_111 = wp::index(var_102, var_66);
        	var_113 = wp::add(var_46, var_112);
        	//wp::store(var_joint_q, var_113, var_111);
        	var_114 = wp::index(var_102, var_68);
        	var_116 = wp::add(var_46, var_115);
        	//wp::store(var_joint_q, var_116, var_114);
        	var_117 = wp::index(var_102, var_60);
        	var_119 = wp::add(var_46, var_118);
        	//wp::store(var_joint_q, var_119, var_117);
        	var_120 = wp::index(var_45, var_9);
        	var_121 = wp::add(var_47, var_9);
        	//wp::store(var_joint_qd, var_121, var_120);
        	var_122 = wp::index(var_45, var_66);
        	var_123 = wp::add(var_47, var_66);
        	//wp::store(var_joint_qd, var_123, var_122);
        	var_124 = wp::index(var_45, var_68);
        	var_125 = wp::add(var_47, var_68);
        	//wp::store(var_joint_qd, var_125, var_124);
        	var_126 = wp::index(var_44, var_9);
        	var_127 = wp::add(var_47, var_60);
        	//wp::store(var_joint_qd, var_127, var_126);
        	var_128 = wp::index(var_44, var_66);
        	var_129 = wp::add(var_47, var_112);
        	//wp::store(var_joint_qd, var_129, var_128);
        	var_130 = wp::index(var_44, var_68);
        	var_131 = wp::add(var_47, var_115);
        	//wp::store(var_joint_qd, var_131, var_130);
        }
        var_132 = wp::select(var_100, var_96, var_102);
        var_134 = (var_37 == var_133);
        if (var_134) {
        	var_135 = wp::transform_get_rotation(var_4);
        	var_136 = wp::quat_inverse(var_135);
        	var_137 = wp::quat_inverse(var_41);
        	var_138 = wp::mul(var_136, var_137);
        	var_139 = wp::mul(var_138, var_42);
        	var_140 = wp::mul(var_139, var_135);
        	var_141 = quat_decompose(var_140);
        	var_144 = wp::vec3(var_142, var_143, var_143);
        	var_145 = wp::index(var_141, var_9);
        	var_146 = wp::quat_from_axis_angle(var_144, var_145);
        	var_147 = wp::vec3(var_143, var_142, var_143);
        	var_148 = wp::quat_rotate(var_146, var_147);
        	var_149 = wp::index(var_141, var_66);
        	var_150 = wp::quat_from_axis_angle(var_148, var_149);
        	var_151 = wp::mul(var_150, var_146);
        	var_152 = wp::vec3(var_143, var_143, var_142);
        	var_153 = wp::quat_rotate(var_151, var_152);
        	var_154 = wp::index(var_141, var_68);
        	var_155 = wp::quat_from_axis_angle(var_153, var_154);
        	var_156 = wp::mul(var_41, var_135);
        	var_157 = wp::index(var_141, var_9);
        	var_158 = wp::add(var_46, var_9);
        	//wp::store(var_joint_q, var_158, var_157);
        	var_159 = wp::index(var_141, var_66);
        	var_160 = wp::add(var_46, var_66);
        	//wp::store(var_joint_q, var_160, var_159);
        	var_161 = wp::index(var_141, var_68);
        	var_162 = wp::add(var_46, var_68);
        	//wp::store(var_joint_q, var_162, var_161);
        	var_163 = wp::quat_rotate(var_156, var_144);
        	var_164 = wp::dot(var_163, var_45);
        	var_165 = wp::add(var_47, var_9);
        	//wp::store(var_joint_qd, var_165, var_164);
        	var_166 = wp::quat_rotate(var_156, var_148);
        	var_167 = wp::dot(var_166, var_45);
        	var_168 = wp::add(var_47, var_66);
        	//wp::store(var_joint_qd, var_168, var_167);
        	var_169 = wp::quat_rotate(var_156, var_153);
        	var_170 = wp::dot(var_169, var_45);
        	var_171 = wp::add(var_47, var_68);
        	//wp::store(var_joint_qd, var_171, var_170);
        	goto label4;
        }
        var_172 = wp::select(var_134, var_132, var_140);
        var_174 = (var_37 == var_173);
        if (var_174) {
        	var_175 = wp::transform_get_rotation(var_4);
        	var_176 = wp::quat_inverse(var_175);
        	var_177 = wp::quat_inverse(var_41);
        	var_178 = wp::mul(var_176, var_177);
        	var_179 = wp::mul(var_178, var_42);
        	var_180 = wp::mul(var_179, var_175);
        	var_181 = quat_decompose(var_180);
        	var_182 = wp::vec3(var_142, var_143, var_143);
        	var_183 = wp::index(var_181, var_9);
        	var_184 = wp::quat_from_axis_angle(var_182, var_183);
        	var_185 = wp::vec3(var_143, var_142, var_143);
        	var_186 = wp::quat_rotate(var_184, var_185);
        	var_187 = wp::index(var_181, var_66);
        	var_188 = wp::quat_from_axis_angle(var_186, var_187);
        	var_189 = wp::mul(var_41, var_175);
        	var_190 = wp::index(var_181, var_9);
        	var_191 = wp::add(var_46, var_9);
        	//wp::store(var_joint_q, var_191, var_190);
        	var_192 = wp::index(var_181, var_66);
        	var_193 = wp::add(var_46, var_66);
        	//wp::store(var_joint_q, var_193, var_192);
        	var_194 = wp::quat_rotate(var_189, var_182);
        	var_195 = wp::dot(var_194, var_45);
        	var_196 = wp::add(var_47, var_9);
        	//wp::store(var_joint_qd, var_196, var_195);
        	var_197 = wp::quat_rotate(var_189, var_186);
        	var_198 = wp::dot(var_197, var_45);
        	var_199 = wp::add(var_47, var_66);
        	//wp::store(var_joint_qd, var_199, var_198);
        	goto label5;
        }
        var_200 = wp::select(var_174, var_172, var_180);
        var_201 = wp::select(var_174, var_135, var_175);
        var_202 = wp::select(var_174, var_141, var_181);
        var_203 = wp::select(var_174, var_144, var_182);
        var_204 = wp::select(var_174, var_146, var_184);
        var_205 = wp::select(var_174, var_148, var_186);
        var_206 = wp::select(var_174, var_150, var_188);
        var_207 = wp::select(var_174, var_156, var_189);
        //---------
        // reverse
        wp::adj_select(var_174, var_156, var_189, adj_174, adj_156, adj_189, adj_207);
        wp::adj_select(var_174, var_150, var_188, adj_174, adj_150, adj_188, adj_206);
        wp::adj_select(var_174, var_148, var_186, adj_174, adj_148, adj_186, adj_205);
        wp::adj_select(var_174, var_146, var_184, adj_174, adj_146, adj_184, adj_204);
        wp::adj_select(var_174, var_144, var_182, adj_174, adj_144, adj_182, adj_203);
        wp::adj_select(var_174, var_141, var_181, adj_174, adj_141, adj_181, adj_202);
        wp::adj_select(var_174, var_135, var_175, adj_174, adj_135, adj_175, adj_201);
        wp::adj_select(var_174, var_172, var_180, adj_174, adj_172, adj_180, adj_200);
        if (var_174) {
        	label5:;
        	wp::adj_store(var_joint_qd, var_199, var_198, adj_joint_qd, adj_199, adj_198);
        	wp::adj_add(var_47, var_66, adj_47, adj_66, adj_199);
        	wp::adj_dot(var_197, var_45, adj_197, adj_45, adj_198);
        	wp::adj_quat_rotate(var_189, var_186, adj_189, adj_186, adj_197);
        	wp::adj_store(var_joint_qd, var_196, var_195, adj_joint_qd, adj_196, adj_195);
        	wp::adj_add(var_47, var_9, adj_47, adj_9, adj_196);
        	wp::adj_dot(var_194, var_45, adj_194, adj_45, adj_195);
        	wp::adj_quat_rotate(var_189, var_182, adj_189, adj_182, adj_194);
        	wp::adj_store(var_joint_q, var_193, var_192, adj_joint_q, adj_193, adj_192);
        	wp::adj_add(var_46, var_66, adj_46, adj_66, adj_193);
        	wp::adj_index(var_181, var_66, adj_181, adj_66, adj_192);
        	wp::adj_store(var_joint_q, var_191, var_190, adj_joint_q, adj_191, adj_190);
        	wp::adj_add(var_46, var_9, adj_46, adj_9, adj_191);
        	wp::adj_index(var_181, var_9, adj_181, adj_9, adj_190);
        	wp::adj_mul(var_41, var_175, adj_41, adj_175, adj_189);
        	wp::adj_quat_from_axis_angle(var_186, var_187, adj_186, adj_187, adj_188);
        	wp::adj_index(var_181, var_66, adj_181, adj_66, adj_187);
        	wp::adj_quat_rotate(var_184, var_185, adj_184, adj_185, adj_186);
        	wp::adj_vec3(var_143, var_142, var_143, adj_143, adj_142, adj_143, adj_185);
        	wp::adj_quat_from_axis_angle(var_182, var_183, adj_182, adj_183, adj_184);
        	wp::adj_index(var_181, var_9, adj_181, adj_9, adj_183);
        	wp::adj_vec3(var_142, var_143, var_143, adj_142, adj_143, adj_143, adj_182);
        	adj_quat_decompose(var_180, adj_180, adj_181);
        	wp::adj_mul(var_179, var_175, adj_179, adj_175, adj_180);
        	wp::adj_mul(var_178, var_42, adj_178, adj_42, adj_179);
        	wp::adj_mul(var_176, var_177, adj_176, adj_177, adj_178);
        	wp::adj_quat_inverse(var_41, adj_41, adj_177);
        	wp::adj_quat_inverse(var_175, adj_175, adj_176);
        	wp::adj_transform_get_rotation(var_4, adj_4, adj_175);
        }
        wp::adj_select(var_134, var_132, var_140, adj_134, adj_132, adj_140, adj_172);
        if (var_134) {
        	label4:;
        	wp::adj_store(var_joint_qd, var_171, var_170, adj_joint_qd, adj_171, adj_170);
        	wp::adj_add(var_47, var_68, adj_47, adj_68, adj_171);
        	wp::adj_dot(var_169, var_45, adj_169, adj_45, adj_170);
        	wp::adj_quat_rotate(var_156, var_153, adj_156, adj_153, adj_169);
        	wp::adj_store(var_joint_qd, var_168, var_167, adj_joint_qd, adj_168, adj_167);
        	wp::adj_add(var_47, var_66, adj_47, adj_66, adj_168);
        	wp::adj_dot(var_166, var_45, adj_166, adj_45, adj_167);
        	wp::adj_quat_rotate(var_156, var_148, adj_156, adj_148, adj_166);
        	wp::adj_store(var_joint_qd, var_165, var_164, adj_joint_qd, adj_165, adj_164);
        	wp::adj_add(var_47, var_9, adj_47, adj_9, adj_165);
        	wp::adj_dot(var_163, var_45, adj_163, adj_45, adj_164);
        	wp::adj_quat_rotate(var_156, var_144, adj_156, adj_144, adj_163);
        	wp::adj_store(var_joint_q, var_162, var_161, adj_joint_q, adj_162, adj_161);
        	wp::adj_add(var_46, var_68, adj_46, adj_68, adj_162);
        	wp::adj_index(var_141, var_68, adj_141, adj_68, adj_161);
        	wp::adj_store(var_joint_q, var_160, var_159, adj_joint_q, adj_160, adj_159);
        	wp::adj_add(var_46, var_66, adj_46, adj_66, adj_160);
        	wp::adj_index(var_141, var_66, adj_141, adj_66, adj_159);
        	wp::adj_store(var_joint_q, var_158, var_157, adj_joint_q, adj_158, adj_157);
        	wp::adj_add(var_46, var_9, adj_46, adj_9, adj_158);
        	wp::adj_index(var_141, var_9, adj_141, adj_9, adj_157);
        	wp::adj_mul(var_41, var_135, adj_41, adj_135, adj_156);
        	wp::adj_quat_from_axis_angle(var_153, var_154, adj_153, adj_154, adj_155);
        	wp::adj_index(var_141, var_68, adj_141, adj_68, adj_154);
        	wp::adj_quat_rotate(var_151, var_152, adj_151, adj_152, adj_153);
        	wp::adj_vec3(var_143, var_143, var_142, adj_143, adj_143, adj_142, adj_152);
        	wp::adj_mul(var_150, var_146, adj_150, adj_146, adj_151);
        	wp::adj_quat_from_axis_angle(var_148, var_149, adj_148, adj_149, adj_150);
        	wp::adj_index(var_141, var_66, adj_141, adj_66, adj_149);
        	wp::adj_quat_rotate(var_146, var_147, adj_146, adj_147, adj_148);
        	wp::adj_vec3(var_143, var_142, var_143, adj_143, adj_142, adj_143, adj_147);
        	wp::adj_quat_from_axis_angle(var_144, var_145, adj_144, adj_145, adj_146);
        	wp::adj_index(var_141, var_9, adj_141, adj_9, adj_145);
        	wp::adj_vec3(var_142, var_143, var_143, adj_142, adj_143, adj_143, adj_144);
        	adj_quat_decompose(var_140, adj_140, adj_141);
        	wp::adj_mul(var_139, var_135, adj_139, adj_135, adj_140);
        	wp::adj_mul(var_138, var_42, adj_138, adj_42, adj_139);
        	wp::adj_mul(var_136, var_137, adj_136, adj_137, adj_138);
        	wp::adj_quat_inverse(var_41, adj_41, adj_137);
        	wp::adj_quat_inverse(var_135, adj_135, adj_136);
        	wp::adj_transform_get_rotation(var_4, adj_4, adj_135);
        }
        wp::adj_select(var_100, var_96, var_102, adj_100, adj_96, adj_102, adj_132);
        if (var_100) {
        	wp::adj_store(var_joint_qd, var_131, var_130, adj_joint_qd, adj_131, adj_130);
        	wp::adj_add(var_47, var_115, adj_47, adj_115, adj_131);
        	wp::adj_index(var_44, var_68, adj_44, adj_68, adj_130);
        	wp::adj_store(var_joint_qd, var_129, var_128, adj_joint_qd, adj_129, adj_128);
        	wp::adj_add(var_47, var_112, adj_47, adj_112, adj_129);
        	wp::adj_index(var_44, var_66, adj_44, adj_66, adj_128);
        	wp::adj_store(var_joint_qd, var_127, var_126, adj_joint_qd, adj_127, adj_126);
        	wp::adj_add(var_47, var_60, adj_47, adj_60, adj_127);
        	wp::adj_index(var_44, var_9, adj_44, adj_9, adj_126);
        	wp::adj_store(var_joint_qd, var_125, var_124, adj_joint_qd, adj_125, adj_124);
        	wp::adj_add(var_47, var_68, adj_47, adj_68, adj_125);
        	wp::adj_index(var_45, var_68, adj_45, adj_68, adj_124);
        	wp::adj_store(var_joint_qd, var_123, var_122, adj_joint_qd, adj_123, adj_122);
        	wp::adj_add(var_47, var_66, adj_47, adj_66, adj_123);
        	wp::adj_index(var_45, var_66, adj_45, adj_66, adj_122);
        	wp::adj_store(var_joint_qd, var_121, var_120, adj_joint_qd, adj_121, adj_120);
        	wp::adj_add(var_47, var_9, adj_47, adj_9, adj_121);
        	wp::adj_index(var_45, var_9, adj_45, adj_9, adj_120);
        	wp::adj_store(var_joint_q, var_119, var_117, adj_joint_q, adj_119, adj_117);
        	wp::adj_add(var_46, var_118, adj_46, adj_118, adj_119);
        	wp::adj_index(var_102, var_60, adj_102, adj_60, adj_117);
        	wp::adj_store(var_joint_q, var_116, var_114, adj_joint_q, adj_116, adj_114);
        	wp::adj_add(var_46, var_115, adj_46, adj_115, adj_116);
        	wp::adj_index(var_102, var_68, adj_102, adj_68, adj_114);
        	wp::adj_store(var_joint_q, var_113, var_111, adj_joint_q, adj_113, adj_111);
        	wp::adj_add(var_46, var_112, adj_46, adj_112, adj_113);
        	wp::adj_index(var_102, var_66, adj_102, adj_66, adj_111);
        	wp::adj_store(var_joint_q, var_110, var_109, adj_joint_q, adj_110, adj_109);
        	wp::adj_add(var_46, var_60, adj_46, adj_60, adj_110);
        	wp::adj_index(var_102, var_9, adj_102, adj_9, adj_109);
        	wp::adj_store(var_joint_q, var_108, var_107, adj_joint_q, adj_108, adj_107);
        	wp::adj_add(var_46, var_68, adj_46, adj_68, adj_108);
        	wp::adj_index(var_43, var_68, adj_43, adj_68, adj_107);
        	wp::adj_store(var_joint_q, var_106, var_105, adj_joint_q, adj_106, adj_105);
        	wp::adj_add(var_46, var_66, adj_46, adj_66, adj_106);
        	wp::adj_index(var_43, var_66, adj_43, adj_66, adj_105);
        	wp::adj_store(var_joint_q, var_104, var_103, adj_joint_q, adj_104, adj_103);
        	wp::adj_add(var_46, var_9, adj_46, adj_9, adj_104);
        	wp::adj_index(var_43, var_9, adj_43, adj_9, adj_103);
        	wp::adj_mul(var_101, var_42, adj_101, adj_42, adj_102);
        	wp::adj_quat_inverse(var_41, adj_41, adj_101);
        }
        if (var_98) {
        	label3:;
        }
        wp::adj_select(var_79, var_58, var_81, adj_79, adj_58, adj_81, adj_96);
        if (var_79) {
        	label2:;
        	wp::adj_store(var_joint_qd, var_95, var_94, adj_joint_qd, adj_95, adj_94);
        	wp::adj_add(var_47, var_68, adj_47, adj_68, adj_95);
        	wp::adj_index(var_45, var_68, adj_45, adj_68, adj_94);
        	wp::adj_store(var_joint_qd, var_93, var_92, adj_joint_qd, adj_93, adj_92);
        	wp::adj_add(var_47, var_66, adj_47, adj_66, adj_93);
        	wp::adj_index(var_45, var_66, adj_45, adj_66, adj_92);
        	wp::adj_store(var_joint_qd, var_91, var_90, adj_joint_qd, adj_91, adj_90);
        	wp::adj_add(var_47, var_9, adj_47, adj_9, adj_91);
        	wp::adj_index(var_45, var_9, adj_45, adj_9, adj_90);
        	wp::adj_store(var_joint_q, var_89, var_88, adj_joint_q, adj_89, adj_88);
        	wp::adj_add(var_46, var_60, adj_46, adj_60, adj_89);
        	wp::adj_index(var_81, var_60, adj_81, adj_60, adj_88);
        	wp::adj_store(var_joint_q, var_87, var_86, adj_joint_q, adj_87, adj_86);
        	wp::adj_add(var_46, var_68, adj_46, adj_68, adj_87);
        	wp::adj_index(var_81, var_68, adj_81, adj_68, adj_86);
        	wp::adj_store(var_joint_q, var_85, var_84, adj_joint_q, adj_85, adj_84);
        	wp::adj_add(var_46, var_66, adj_46, adj_66, adj_85);
        	wp::adj_index(var_81, var_66, adj_81, adj_66, adj_84);
        	wp::adj_store(var_joint_q, var_83, var_82, adj_joint_q, adj_83, adj_82);
        	wp::adj_add(var_46, var_9, adj_46, adj_9, adj_83);
        	wp::adj_index(var_81, var_9, adj_81, adj_9, adj_82);
        	wp::adj_mul(var_80, var_42, adj_80, adj_42, adj_81);
        	wp::adj_quat_inverse(var_41, adj_41, adj_80);
        }
        wp::adj_select(var_54, var_52, var_74, adj_54, adj_52, adj_74, adj_77);
        wp::adj_select(var_54, var_51, var_73, adj_54, adj_51, adj_73, adj_76);
        wp::adj_select(var_54, var_50, var_55, adj_54, adj_50, adj_55, adj_75);
        if (var_54) {
        	label1:;
        	wp::adj_store(var_joint_qd, var_47, var_74, adj_joint_qd, adj_47, adj_74);
        	wp::adj_store(var_joint_q, var_46, var_73, adj_joint_q, adj_46, adj_73);
        	wp::adj_dot(var_45, var_55, adj_45, adj_55, adj_74);
        	wp::adj_mul(var_64, var_72, adj_64, adj_72, adj_73);
        	wp::adj_sign(var_71, adj_71, adj_72);
        	wp::adj_dot(var_38, var_70, adj_38, adj_70, adj_71);
        	wp::adj_vec3(var_65, var_67, var_69, adj_65, adj_67, adj_69, adj_70);
        	wp::adj_index(var_59, var_68, adj_59, adj_68, adj_69);
        	wp::adj_index(var_59, var_66, adj_59, adj_66, adj_67);
        	wp::adj_index(var_59, var_9, adj_59, adj_9, adj_65);
        	wp::adj_mul(var_62, var_63, adj_62, adj_63, adj_64);
        	wp::adj_acos(var_61, adj_61, adj_62);
        	wp::adj_index(var_59, var_60, adj_59, adj_60, adj_61);
        	adj_quat_twist(var_38, var_58, adj_38, adj_58, adj_59);
        	wp::adj_mul(var_57, var_42, adj_57, adj_42, adj_58);
        	wp::adj_quat_inverse(var_41, adj_41, adj_57);
        	wp::adj_transform_vector(var_26, var_38, adj_26, adj_38, adj_56);
        	wp::adj_transform_vector(var_23, var_38, adj_23, adj_38, adj_55);
        }
        if (var_49) {
        	label0:;
        	wp::adj_store(var_joint_qd, var_47, var_52, adj_joint_qd, adj_47, adj_52);
        	wp::adj_store(var_joint_q, var_46, var_51, adj_joint_q, adj_46, adj_51);
        	wp::adj_dot(var_44, var_50, adj_44, adj_50, adj_52);
        	wp::adj_dot(var_43, var_50, adj_43, adj_50, adj_51);
        	wp::adj_transform_vector(var_23, var_38, adj_23, adj_38, adj_50);
        }
        wp::adj_load(var_joint_qd_start, var_0, adj_joint_qd_start, adj_0, adj_47);
        wp::adj_load(var_joint_q_start, var_0, adj_joint_q_start, adj_0, adj_46);
        wp::adj_sub(var_33, var_24, adj_33, adj_24, adj_45);
        wp::adj_sub(var_36, var_25, adj_36, adj_25, adj_44);
        wp::adj_sub(var_40, var_39, adj_40, adj_39, adj_43);
        wp::adj_transform_get_rotation(var_26, adj_26, adj_42);
        wp::adj_transform_get_rotation(var_23, adj_23, adj_41);
        wp::adj_transform_get_translation(var_26, adj_26, adj_40);
        wp::adj_transform_get_translation(var_23, adj_23, adj_39);
        wp::adj_load(var_joint_axis, var_0, adj_joint_axis, adj_0, adj_38);
        wp::adj_load(var_joint_type, var_0, adj_joint_type, adj_0, adj_37);
        wp::adj_add(var_34, var_35, adj_34, adj_35, adj_36);
        wp::adj_cross(var_33, var_31, adj_33, adj_31, adj_35);
        wp::adj_spatial_bottom(var_32, adj_32, adj_34);
        wp::adj_spatial_top(var_32, adj_32, adj_33);
        wp::adj_load(var_body_qd, var_1, adj_body_qd, adj_1, adj_32);
        wp::adj_sub(var_27, var_30, adj_27, adj_30, adj_31);
        wp::adj_transform_point(var_28, var_29, adj_28, adj_29, adj_30);
        wp::adj_load(var_body_com, var_1, adj_body_com, adj_1, adj_29);
        wp::adj_load(var_body_q, var_1, adj_body_q, adj_1, adj_28);
        wp::adj_transform_get_translation(var_26, adj_26, adj_27);
        wp::adj_load(var_body_q, var_1, adj_body_q, adj_1, adj_26);
        wp::adj_select(var_10, var_8, var_22, adj_10, adj_8, adj_22, adj_25);
        wp::adj_select(var_10, var_7, var_19, adj_10, adj_7, adj_19, adj_24);
        wp::adj_select(var_10, var_5, var_12, adj_10, adj_5, adj_12, adj_23);
        if (var_10) {
        	wp::adj_add(var_20, var_21, adj_20, adj_21, adj_22);
        	wp::adj_cross(var_19, var_17, adj_19, adj_17, adj_21);
        	wp::adj_spatial_bottom(var_18, adj_18, adj_20);
        	wp::adj_spatial_top(var_18, adj_18, adj_19);
        	wp::adj_load(var_body_qd, var_2, adj_body_qd, adj_2, adj_18);
        	wp::adj_sub(var_13, var_16, adj_13, adj_16, adj_17);
        	wp::adj_transform_point(var_14, var_15, adj_14, adj_15, adj_16);
        	wp::adj_load(var_body_com, var_2, adj_body_com, adj_2, adj_15);
        	wp::adj_load(var_body_q, var_2, adj_body_q, adj_2, adj_14);
        	wp::adj_transform_get_translation(var_12, adj_12, adj_13);
        	wp::adj_mul(var_11, var_5, adj_11, adj_5, adj_12);
        	wp::adj_load(var_body_q, var_2, adj_body_q, adj_2, adj_11);
        }
        wp::adj_copy(var_5, var_3, adj_5, adj_3);
        wp::adj_load(var_joint_X_c, var_0, adj_joint_X_c, adj_0, adj_4);
        wp::adj_load(var_joint_X_p, var_0, adj_joint_X_p, adj_0, adj_3);
        wp::adj_load(var_joint_parent, var_0, adj_joint_parent, adj_0, adj_2);
        wp::adj_copy(var_1, var_0, adj_1, adj_0);
        return;

}



extern "C" {

// Python entry points
WP_API void eval_articulation_ik_cuda_forward(void* stream, launch_bounds_t dim,
	array_t<transform> var_body_q,
	array_t<spatial_vector> var_body_qd,
	array_t<vec3> var_body_com,
	array_t<int32> var_joint_type,
	array_t<int32> var_joint_parent,
	array_t<transform> var_joint_X_p,
	array_t<transform> var_joint_X_c,
	array_t<vec3> var_joint_axis,
	array_t<int32> var_joint_q_start,
	array_t<int32> var_joint_qd_start,
	array_t<float32> var_joint_q,
	array_t<float32> var_joint_qd)
{
    eval_articulation_ik_cuda_kernel_forward<<<(dim.size + 256 - 1) / 256, 256, 0, (cudaStream_t)stream>>>(dim,
			var_body_q,
			var_body_qd,
			var_body_com,
			var_joint_type,
			var_joint_parent,
			var_joint_X_p,
			var_joint_X_c,
			var_joint_axis,
			var_joint_q_start,
			var_joint_qd_start,
			var_joint_q,
			var_joint_qd);
}

WP_API void eval_articulation_ik_cuda_backward(void* stream, launch_bounds_t dim,
	array_t<transform> var_body_q,
	array_t<spatial_vector> var_body_qd,
	array_t<vec3> var_body_com,
	array_t<int32> var_joint_type,
	array_t<int32> var_joint_parent,
	array_t<transform> var_joint_X_p,
	array_t<transform> var_joint_X_c,
	array_t<vec3> var_joint_axis,
	array_t<int32> var_joint_q_start,
	array_t<int32> var_joint_qd_start,
	array_t<float32> var_joint_q,
	array_t<float32> var_joint_qd,
	array_t<transform> adj_body_q,
	array_t<spatial_vector> adj_body_qd,
	array_t<vec3> adj_body_com,
	array_t<int32> adj_joint_type,
	array_t<int32> adj_joint_parent,
	array_t<transform> adj_joint_X_p,
	array_t<transform> adj_joint_X_c,
	array_t<vec3> adj_joint_axis,
	array_t<int32> adj_joint_q_start,
	array_t<int32> adj_joint_qd_start,
	array_t<float32> adj_joint_q,
	array_t<float32> adj_joint_qd)
{
    eval_articulation_ik_cuda_kernel_backward<<<(dim.size + 256 - 1) / 256, 256, 0, (cudaStream_t)stream>>>(dim,
			var_body_q,
			var_body_qd,
			var_body_com,
			var_joint_type,
			var_joint_parent,
			var_joint_X_p,
			var_joint_X_c,
			var_joint_axis,
			var_joint_q_start,
			var_joint_qd_start,
			var_joint_q,
			var_joint_qd,
			adj_body_q,
			adj_body_qd,
			adj_body_com,
			adj_joint_type,
			adj_joint_parent,
			adj_joint_X_p,
			adj_joint_X_c,
			adj_joint_axis,
			adj_joint_q_start,
			adj_joint_qd_start,
			adj_joint_q,
			adj_joint_qd);
}

} // extern C

