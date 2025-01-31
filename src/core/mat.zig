const c = @import("../c_api.zig");
const std = @import("std");
const utils = @import("../utils.zig");
const core = @import("../core.zig");
const assert = std.debug.assert;
const epnn = utils.ensurePtrNotNull;
const Rect = core.Rect;
const Color = core.Color;
const Scalar = core.Scalar;
const Point = core.Point;
const PointVector = core.PointVector;
const RNG = core.RNG;
const BorderType = core.BorderType;
const TermCriteria = core.TermCriteria;
const NormType = core.NormType;

ptr: CSelf,

const Self = @This();
const CSelf = c.Mat;

const OperationType = enum {
    add,
    subtract,
    multiply,
    divide,
};

const BitOperationType = enum {
    and_,
    or_,
    xor_,
    not_,
};

// TODO:apply packed struct like this
// pub const MatType = packed struct {
//     type: u3,
//     channel: u2,
// };
pub const MatType = enum(u5) {
    // MatTypeCV8UC1 is a Mat of 8-bit unsigned int with a single channel
    cv8uc1 = cv8u + channels1,

    // MatTypeCV8UC2 is a Mat of 8-bit unsigned int with 2 channels
    cv8uc2 = cv8u + channels2,

    // MatTypeCV8UC3 is a Mat of 8-bit unsigned int with 3 channels
    cv8uc3 = cv8u + channels3,

    // MatTypeCV8UC4 is a Mat of 8-bit unsigned int with 4 channels
    cv8uc4 = cv8u + channels4,

    // MatTypeCV8SC1 is a Mat of 8-bit signed int with a single channel
    cv8sc1 = cv8s + channels1,

    // MatTypeCV8SC2 is a Mat of 8-bit signed int with 2 channels
    cv8sc2 = cv8s + channels2,

    // MatTypeCV8SC3 is a Mat of 8-bit signed int with 3 channels
    cv8sc3 = cv8s + channels3,

    // MatTypeCV8SC4 is a Mat of 8-bit signed int with 4 channels
    cv8sc4 = cv8s + channels4,

    // MatTypeCV16UC1 is a Mat of 16-bit unsigned int with a single channel
    cv16uc1 = cv16u + channels1,

    // MatTypeCV16UC2 is a Mat of 16-bit unsigned int with 2 channels
    cv16uc2 = cv16u + channels2,

    // MatTypeCV16UC3 is a Mat of 16-bit unsigned int with 3 channels
    cv16uc3 = cv16u + channels3,

    // MatTypeCV16UC4 is a Mat of 16-bit unsigned int with 4 channels
    cv16uc4 = cv16u + channels4,

    // MatTypeCV16SC1 is a Mat of 16-bit signed int with a single channel
    cv16sc1 = cv16s + channels1,

    // MatTypeCV16SC2 is a Mat of 16-bit signed int with 2 channels
    cv16sc2 = cv16s + channels2,

    // MatTypeCV16SC3 is a Mat of 16-bit signed int with 3 channels
    cv16sc3 = cv16s + channels3,

    // MatTypeCV16SC4 is a Mat of 16-bit signed int with 4 channels
    cv16sc4 = cv16s + channels4,

    // MatTypeCV32SC1 is a Mat of 32-bit signed int with a single channel
    cv32sc1 = cv32s + channels1,

    // MatTypeCV32SC2 is a Mat of 32-bit signed int with 2 channels
    cv32sc2 = cv32s + channels2,

    // MatTypeCV32SC3 is a Mat of 32-bit signed int with 3 channels
    cv32sc3 = cv32s + channels3,

    // MatTypeCV32SC4 is a Mat of 32-bit signed int with 4 channels
    cv32sc4 = cv32s + channels4,

    // MatTypeCV32FC1 is a Mat of 32-bit float int with a single channel
    cv32fc1 = cv32f + channels1,

    // MatTypeCV32FC2 is a Mat of 32-bit float int with 2 channels
    cv32fc2 = cv32f + channels2,

    // MatTypeCV32FC3 is a Mat of 32-bit float int with 3 channels
    cv32fc3 = cv32f + channels3,

    // MatTypeCV32FC4 is a Mat of 32-bit float int with 4 channels
    cv32fc4 = cv32f + channels4,

    // MatTypeCV64FC1 is a Mat of 64-bit float int with a single channel
    cv64fc1 = cv64f + channels1,

    // MatTypeCV64FC2 is a Mat of 64-bit float int with 2 channels
    cv64fc2 = cv64f + channels2,

    // MatTypeCV64FC3 is a Mat of 64-bit float int with 3 channels
    cv64fc3 = cv64f + channels3,

    // MatTypeCV64FC4 is a Mat of 64-bit float int with 4 channels
    cv64fc4 = cv64f + channels4,

    const channels1 = 0 << 3;
    const channels2 = 1 << 3;
    const channels3 = 2 << 3;
    const channels4 = 3 << 3;
    // MatTypeCV8U is a Mat of 8-bit unsigned int
    const cv8u = 0;
    // MatTypeCV8S is a Mat of 8-bit signed int
    const cv8s = 1;
    // MatTypeCV16U is a Mat of 16-bit unsigned int
    const cv16u = 2;
    // MatTypeCV16S is a Mat of 16-bit signed int
    const cv16s = 3;
    // MatTypeCV32S is a Mat of 32-bit signed int
    const cv32s = 4;
    // MatTypeCV32F is a Mat of 32-bit float
    const cv32f = 5;
    // MatTypeCV64F is a Mat of 64-bit float
    const cv64f = 6;
};

pub const CompareType = enum(u3) {
    // CompareEQ src1 is equal to src2.
    eq = 0,

    // CompareGT src1 is greater than src2.
    gt = 1,

    // CompareGE src1 is greater than or equal to src2.
    ge = 2,

    // CompareLT src1 is less than src2.
    lt = 3,

    // CompareLE src1 is less than or equal to src2.
    le = 4,

    // CompareNE src1 is unequal to src2.
    ne = 5,
};

pub const SolveDecompositionFlag = enum(u3) {
    // Gaussian elimination with the optimal pivot element chosen.
    lu = 0,

    // Singular value decomposition (SVD) method. The system can be over-defined and/or the matrix src1 can be singular.
    svd = 1,

    // Eigenvalue decomposition. The matrix src1 must be symmetrical.
    eing = 2,

    // Cholesky LL^T factorization. The matrix src1 must be symmetrical and positively defined.
    cholesky = 3,

    // QR factorization. The system can be over-defined and/or the matrix src1 can be singular.
    qr = 4,

    // While all the previous flags are mutually exclusive, this flag can be used together with any of the previous.
    // It means that the normal equations 𝚜𝚛𝚌𝟷^T⋅𝚜𝚛𝚌𝟷⋅𝚍𝚜𝚝=𝚜𝚛𝚌𝟷^T𝚜𝚛𝚌𝟸 are solved instead of the original system
    // 𝚜𝚛𝚌𝟷⋅𝚍𝚜𝚝=𝚜𝚛𝚌𝟸.
    normal = 5,
};

pub const ReduceType = enum(u3) {
    /// The output is the sum of all rows/columns of the matrix.
    sum = 0,

    /// The output is the mean vector of all rows/columns of the matrix.
    avg = 1,

    /// The output is the maximum (column/row-wise) of all rows/columns of the matrix.
    max = 2,

    /// The output is the minimum (column/row-wise) of all rows/columns of the matrix.
    min = 3,
};

/// CovarFlags are the covariation flags used by functions such as BorderInterpolate.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d0/de1/group__core.html#ga719ebd4a73f30f4fab258ab7616d0f0f
///
pub const CovarFlags = struct {
    type: enum(u1) {
        /// CovarScrambled indicates to scramble the results.
        scrambled = 0,

        /// CovarNormal indicates to use normal covariation.
        normal = 1,
    },

    /// CovarUseAvg indicates to use average covariation.
    use_avg: bool = false,

    /// CovarScale indicates to use scaled covariation.
    scale: bool = false,

    /// CovarRows indicates to use covariation on rows.
    rows: bool = false,

    /// CovarCols indicates to use covariation on columns.
    cols: bool = false,

    pub fn toNum(self: CovarFlags) u32 {
        return @as(u5, @bitCast(packed struct {
            type: u1,
            use_avg: bool,
            scale: bool,
            rows: bool,
            cols: bool,
        }{
            .type = @intFromEnum(self.type),
            .use_avg = self.use_avg,
            .scale = self.scale,
            .rows = self.rows,
            .cols = self.cols,
        }));
    }

    comptime {
        std.debug.assert((CovarFlags{ .type = .scrambled }).toNum() == 0);
        std.debug.assert((CovarFlags{ .type = .normal }).toNum() == 1);
        std.debug.assert((CovarFlags{ .type = .scrambled, .use_avg = true }).toNum() == 2);
        std.debug.assert((CovarFlags{ .type = .scrambled, .scale = true }).toNum() == 4);
        std.debug.assert((CovarFlags{ .type = .scrambled, .rows = true }).toNum() == 8);
        std.debug.assert((CovarFlags{ .type = .scrambled, .cols = true }).toNum() == 16);
    }
};

/// DftFlags represents a DFT or DCT flag.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaf4dde112b483b38175621befedda1f1c
///
pub const DftFlags = struct {
    ///  if false, the function does a forward 1D or 2D transform. Otherwise, it is an inverse 1D or 2D transform.
    inverse: bool = false,

    /// DftScale scales the result: divide it by the number of array elements. Normally, it is combined with DFT_INVERSE.
    scale: bool = false,

    /// if true, the function performs a 1D transform of each row.
    rows: bool = false,

    /// DftComplexOutput performs a forward transformation of 1D or 2D real array; the result, though being a complex array, has complex-conjugate symmetry
    complex_output: bool = false,

    /// DftRealOutput performs an inverse transformation of a 1D or 2D complex array; the result is normally a complex array of the same size,
    /// however, if the input array has conjugate-complex symmetry (for example, it is a result of forward transformation with DFT_COMPLEX_OUTPUT flag),
    /// the output is a real array.
    real_output: bool = false,

    /// DftComplexInput specifies that input is complex input. If this flag is set, the input must have 2 channels.
    complex_input: bool = false,

    pub fn toNum(self: DftFlags) u7 {
        return @as(u7, @bitCast(packed struct {
            inverse: bool,
            scale: bool,
            rows: bool,
            _padding: u1 = 0,
            complex_output: bool,
            real_output: bool,
            complex_input: bool,
        }{
            .inverse = self.inverse,
            .scale = self.scale,
            .rows = self.rows,
            .complex_output = self.complex_output,
            .real_output = self.real_output,
            .complex_input = self.complex_input,
        }));
    }

    comptime {
        std.debug.assert((DftFlags{ .inverse = true }).toNum() == 1);
        std.debug.assert((DftFlags{ .scale = true }).toNum() == 2);
        std.debug.assert((DftFlags{ .rows = true }).toNum() == 4);
        std.debug.assert((DftFlags{ .complex_output = true }).toNum() == 16);
        std.debug.assert((DftFlags{ .real_output = true }).toNum() == 32);
        std.debug.assert((DftFlags{ .complex_input = true }).toNum() == 64);
    }
};

pub const GemmFlags = enum(u3) {
    /// Gemm1T indicates to transpose the first matrix.
    gemm_1t = 1,

    /// Gemm2T indicates to transpose the second matrix.
    gemm_2t = 2,

    /// Gemm3T indicates to transpose the third matrix.
    gemm_3t = 4,
};

// RotateFlag for image rotation
//
//
// For further details please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6f45d55c0b1cc9d97f5353a7c8a7aac2
pub const RotateFlag = enum(u2) {
    /// Rotate90Clockwise allows to rotate image 90 degrees clockwise
    rotate_90_clockwise = 0,
    /// Rotate180Clockwise allows to rotate image 180 degrees clockwise
    rotate_180_clockwise = 1,
    /// Rotate90CounterClockwise allows to rotate 270 degrees clockwise
    rotate_90_counter_clockwise = 2,
};

// KMeansFlag for kmeans center selection
//
// For further details, please see:
// https://docs.opencv.org/master/d0/de1/group__core.html#ga276000efe55ee2756e0c471c7b270949
pub const KMeansFlag = enum(u2) {
    // KMeansRandomCenters selects random initial centers in each attempt.
    random_centers = 0,
    // KMeansPPCenters uses kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007].
    pp_centers = 1,
    // KMeansUseInitialLabels uses the user-supplied lables during the first (and possibly the only) attempt
    // instead of computing them from the initial centers. For the second and further attempts, use the random or semi-random     // centers. Use one of KMEANS_*_CENTERS flag to specify the exact method.
    use_initial_labels = 2,
};

pub const SortFlags = struct {
    type: enum(u1) {
        /// Each matrix row is sorted independently
        row = 0,

        /// Each matrix column is sorted independently;
        column = 1,
    } = .row,
    descending: bool = false,

    pub fn toNum(self: SortFlags) u5 {
        return @as(u5, @bitCast(packed struct {
            type: u1,
            _padding: u3 = 0,
            descending: bool,
        }{
            .type = @intFromEnum(self.type),
            .descending = self.descending,
        }));
    }

    comptime {
        std.debug.assert((SortFlags{ .type = .column }).toNum() == 1);
        std.debug.assert((SortFlags{ .descending = true }).toNum() == 16);
    }
};

pub fn initFromC(ptr: CSelf) !Self {
    const nn_ptr = try epnn(ptr);
    return Self{ .ptr = nn_ptr };
}

/// init Mat
pub fn init() !Self {
    const ptr = c.Mat_New();
    return try Self.initFromC(ptr);
}

/// init Mat with size and type
pub fn initSize(n_rows: i32, n_cols: i32, mt: MatType) !Self {
    const ptr = c.Mat_NewWithSize(n_rows, n_cols, @intFromEnum(mt));
    return try Self.initFromC(ptr);
}

/// init multidimentional Mat with sizes and type
pub fn initSizes(size_array: []i32, mt: MatType) !Self {
    const c_size_vector = c.IntVector{
        .val = @as([*]i32, @ptrCast(size_array)),
        .length = @as(i32, @intCast(size_array.len)),
    };

    const ptr = c.Mat_NewWithSizes(c_size_vector, @intFromEnum(mt));
    return try Self.initFromC(ptr);
}

pub fn initFromMat(self: *Self, n_rows: i32, n_cols: i32, mt: MatType, prows: i32, pcols: i32) !Self {
    const mat_ptr = try epnn(self.*.ptr);
    const ptr = c.Mat_FromPtr(mat_ptr, n_rows, n_cols, @intFromEnum(mt), prows, pcols);
    return try Self.initFromC(ptr);
}

pub fn initFromScalar(s: Scalar) !Self {
    const ptr = c.Mat_NewFromScalar(s.toC());
    return try Self.initFromC(ptr);
}

pub fn initFromBytes(rows_: i32, cols_: i32, bytes: []u8, mt: MatType) !Self {
    const c_bytes = c.ByteVector{
        .val = @as([*]u8, @ptrCast(bytes)),
        .length = @as(i32, @intCast(bytes.len)),
    };
    const ptr = c.Mat_NewFromBytes(rows_, cols_, @intFromEnum(mt), c_bytes);
    return try Self.initFromC(ptr);
}

pub fn initSizeFromScalar(s: Scalar, n_rows: i32, n_cols: i32, mt: MatType) !Self {
    const ptr = c.Mat_NewWithSizeFromScalar(s.toC(), n_rows, n_cols, @intFromEnum(mt));
    return try Self.initFromC(ptr);
}

pub fn initSizesFromScalar(size_array: []const i32, s: Scalar, mt: MatType) !Self {
    const c_size_vector = c.IntVector{
        .val = @as([*]i32, @ptrCast(size_array)),
        .length = @as(i32, @intCast(size_array.len)),
    };
    const ptr = c.Mat_NewWithSizesFromScalar(c_size_vector, @intFromEnum(mt), s.toC());
    return try Self.initFromC(ptr);
}

pub fn initSizesFromBytes(size_array: []const i32, bytes: []u8, mt: MatType) !Self {
    const c_size_vector = c.IntVector{
        .val = @as([*]i32, @ptrCast(size_array)),
        .length = @as(i32, @intCast(size_array.len)),
    };
    const c_bytes = c.ByteVector{
        .val = @as([*]u8, @ptrCast(bytes)),
        .length = @as(i32, @intCast(bytes.len)),
    };
    const ptr = c.Mat_NewWithSizesFromBytes(c_size_vector, @intFromEnum(mt), c_bytes);
    return try Self.initFromC(ptr);
}

/// Returns an identity matrix of the specified size and type.
///
/// The method returns a Matlab-style identity matrix initializer, similarly to Mat::zeros. Similarly to Mat::ones.
/// For further details, please see:
/// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a2cf9b9acde7a9852542bbc20ef851ed2
pub fn initEye(rows_: i32, cols_: i32, mt: MatType) !Self {
    const ptr = c.Eye(rows_, cols_, @intFromEnum(mt));
    return try Self.initFromC(ptr);
}

/// Returns a zero array of the specified size and type.
///
/// The method returns a Matlab-style zero array initializer.
/// For further details, please see:
/// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a0b57b6a326c8876d944d188a46e0f556
pub fn initZeros(rows_: i32, cols_: i32, mt: MatType) !Self {
    const ptr = c.Zeros(rows_, cols_, @intFromEnum(mt));
    return try Self.initFromC(ptr);
}

/// Returns an array of all 1's of the specified size and type.
///
/// The method returns a Matlab-style 1's array initializer
/// For further details, please see:
/// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a69ae0402d116fc9c71908d8508dc2f09
pub fn initOnes(rows_: i32, cols_: i32, mt: MatType) !Self {
    const ptr = c.Ones(rows_, cols_, @intFromEnum(mt));
    return try Self.initFromC(ptr);
}

pub fn deinit(self: *Self) void {
    assert(self.*.ptr != null);
    c.Mat_Close(self.ptr);
    self.ptr = null;
}

pub fn toC(self: Self) CSelf {
    return self.ptr;
}

/// CopyTo copies Mat into destination Mat.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a33fd5d125b4c302b0c9aa86980791a77
///
pub fn copyTo(self: Self, dest: *Self) void {
    _ = c.Mat_CopyTo(self.ptr, dest.*.ptr);
}

/// CopyToWithMask copies Mat into destination Mat after applying the mask Mat.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a626fe5f96d02525e2604d2ad46dd574f
///
pub fn copyToWithMask(self: Self, dest: *Self, mask: Self) void {
    _ = c.Mat_CopyToWithMask(self.ptr, dest.*.ptr, mask.ptr);
}

pub fn clone(self: Self) !Self {
    const ptr = c.Mat_Clone(self.ptr);
    return try Self.initFromC(ptr);
}

// ConvertTo converts Mat into destination Mat.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#adf88c60c5b4980e05bb556080916978b
//
pub fn convertTo(self: Self, dst: *Self, mt: MatType) void {
    _ = c.Mat_ConvertTo(self.ptr, dst.*.ptr, @intFromEnum(mt));
}
pub fn convertToWithParams(self: Self, dst: *Self, mt: MatType, alpha: f32, beta: f32) void {
    _ = c.Mat_ConvertToWithParams(self.ptr, dst.*.ptr, @intFromEnum(mt), alpha, beta);
}

pub fn cols(self: Self) i32 {
    return c.Mat_Cols(self.ptr);
}

pub fn rows(self: Self) i32 {
    return c.Mat_Rows(self.ptr);
}

pub fn channels(self: Self) i32 {
    return c.Mat_Channels(self.ptr);
}

pub fn getType(self: Self) MatType {
    const type_ = c.Mat_Type(self.ptr);
    return @as(MatType, @enumFromInt(type_));
}

pub fn step(self: Self) i32 {
    return c.Mat_Step(self.ptr);
}

pub fn elemSize(self: Self) i32 {
    return c.Mat_ElemSize(self.ptr);
}

// Total returns the total number of array elements.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#aa4d317d43fb0cba9c2503f3c61b866c8
//
pub fn total(self: Self) i32 {
    return c.Mat_Total(self.ptr);
}

// Size returns an array with one element for each dimension containing the size of that dimension for the Mat.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#aa4d317d43fb0cba9c2503f3c61b866c8
//
pub fn size(self: Self) []const i32 {
    var v: c.IntVector = undefined;
    _ = c.Mat_Size(self.ptr, &v);
    return v.val[0..@as(usize, @intCast(v.length))];
}

// getAt returns a value from a specific row/col
// in this Mat expecting it to be of type uchar aka CV_8U.
// in this Mat expecting it to be of type schar aka CV_8S.
// in this Mat expecting it to be of type short aka CV_16S.
// in this Mat expecting it to be of type int aka CV_32S.
// in this Mat expecting it to be of type float aka CV_32F.
// in this Mat expecting it to be of type double aka CV_64F.
pub fn get(self: Self, comptime T: type, row: usize, col: usize) T {
    const row_ = @as(i32, @intCast(row));
    const col_ = @as(i32, @intCast(col));
    return switch (T) {
        u8 => c.Mat_GetUChar(self.ptr, row_, col_),
        i8 => c.Mat_GetSChar(self.ptr, row_, col_),
        i16 => c.Mat_GetShort(self.ptr, row_, col_),
        i32 => c.Mat_GetInt(self.ptr, row_, col_),
        f32 => c.Mat_GetFloat(self.ptr, row_, col_),
        f64 => c.Mat_GetDouble(self.ptr, row_, col_),
        else => @compileError("not implemented for " ++ @typeName(T)),
    };
}

// getAt3 returns a value from a specific x, y, z coordinate location
// in this Mat expecting it to be of type uchar aka CV_8U.
// in this Mat expecting it to be of type schar aka CV_8S.
// in this Mat expecting it to be of type short aka CV_16S.
// in this Mat expecting it to be of type int aka CV_32S.
// in this Mat expecting it to be of type float aka CV_32F.
// in this Mat expecting it to be of type double aka CV_64F.
pub fn get3(self: Self, comptime T: type, x: usize, y: usize, z: usize) T {
    const x_ = @as(i32, @intCast(x));
    const y_ = @as(i32, @intCast(y));
    const z_ = @as(i32, @intCast(z));
    return switch (T) {
        u8 => c.Mat_GetUChar3(self.ptr, x_, y_, z_),
        i8 => c.Mat_GetSChar3(self.ptr, x_, y_, z_),
        i16 => c.Mat_GetShort3(self.ptr, x_, y_, z_),
        i32 => c.Mat_GetInt3(self.ptr, x_, y_, z_),
        f32 => c.Mat_GetFloat3(self.ptr, x_, y_, z_),
        f64 => c.Mat_GetDouble3(self.ptr, x_, y_, z_),
        else => @compileError("not implemented for " ++ @typeName(T)),
    };
}

pub fn setTo(self: *Self, value: Scalar) void {
    _ = c.Mat_SetTo(self.ptr, value.toC());
}

pub fn set(self: *Self, comptime T: type, row: usize, col: usize, val: T) void {
    const row_ = @as(u31, @intCast(row));
    const col_ = @as(u31, @intCast(col));
    _ = switch (T) {
        u8 => c.Mat_SetUChar(self.ptr, row_, col_, val),
        i8 => c.Mat_SetSChar(self.ptr, row_, col_, val),
        i16 => c.Mat_SetShort(self.ptr, row_, col_, val),
        i32 => c.Mat_SetInt(self.ptr, row_, col_, val),
        f32 => c.Mat_SetFloat(self.ptr, row_, col_, val),
        f64 => c.Mat_SetDouble(self.ptr, row_, col_, val),
        else => @compileError("not implemented for " ++ @typeName(T)),
    };
}

pub fn set3(self: *Self, comptime T: type, x: usize, y: usize, z: usize, val: T) void {
    const x_ = @as(u31, @intCast(x));
    const y_ = @as(u31, @intCast(y));
    const z_ = @as(u31, @intCast(z));
    _ = switch (T) {
        u8 => c.Mat_SetUChar3(self.ptr, x_, y_, z_, val),
        i8 => c.Mat_SetSChar3(self.ptr, x_, y_, z_, val),
        i16 => c.Mat_SetShort3(self.ptr, x_, y_, z_, val),
        i32 => c.Mat_SetInt3(self.ptr, x_, y_, z_, val),
        f32 => c.Mat_SetFloat3(self.ptr, x_, y_, z_, val),
        f64 => c.Mat_SetDouble3(self.ptr, x_, y_, z_, val),
        else => @compileError("not implemented for " ++ @typeName(T)),
    };
}

// IsContinuous determines if the Mat is continuous.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#aa90cea495029c7d1ee0a41361ccecdf3
//
pub fn isContinuous(self: Self) bool {
    return c.Mat_IsContinuous(self.ptr);
}

pub fn isEmpty(self: Self) bool {
    return c.Mat_Empty(self.ptr) != 0;
}

// Sqrt calculates a square root of array elements.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga186222c3919657890f88df5a1f64a7d7
//
pub fn sqrt(self: Self) !Self {
    const ptr = c.Mat_Sqrt(self.ptr);
    return try initFromC(ptr);
}

// Mean calculates the mean value M of array elements, independently for each channel, and return it as Scalar
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga191389f8a0e58180bb13a727782cd461
//
pub fn mean(self: Self) Scalar {
    return Scalar.fromC(c.Mat_Mean(self.ptr));
}

// ConvertFp16 converts a Mat to half-precision floating point.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga9c25d9ef44a2a48ecc3774b30cb80082
//
pub fn convertFp16(self: Self) Self {
    return .{ .ptr = c.Mat_ConvertFp16(self.ptr) };
}

// MeanWithMask calculates the mean value M of array elements,independently for each channel,
// and returns it as Scalar vector while applying the mask.
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga191389f8a0e58180bb13a727782cd461
//
pub fn meanWithMask(self: Self, mask: Self) Scalar {
    return Scalar.initFromC(c.Mat_MeanWithMask(self.ptr, mask.ptr));
}

pub fn calcValueInplace(self: *Self, comptime op: OperationType, v: anytype) void {
    const T = @TypeOf(v);
    _ = switch (T) {
        u8 => switch (op) {
            .add => c.Mat_AddUChar(self.ptr, v, op),
            .subtract => c.Mat_SubtractUChar(self.ptr, v, op),
            .multiply => c.Mat_MultiplyUChar(self.ptr, v, op),
            .divide => c.Mat_DivideUChar(self.ptr, v, op),
        },
        f32 => switch (op) {
            .add => c.Mat_AddFloat(self.ptr, v, op),
            .subtract => c.Mat_SubtractFloat(self.ptr, v, op),
            .multiply => c.Mat_MultiplyFloat(self.ptr, v, op),
            .divide => c.Mat_DivideFloat(self.ptr, v, op),
        },
        else => @compileError("not implemented for " ++ @typeName(T)),
    };
}

pub fn addValueInplace(self: *Self, v: anytype) void {
    self.calcValueInplace(.add, v);
}

pub fn subtractValueInplace(self: *Self, v: anytype) void {
    self.calcValueInplace(.subtract, v);
}

pub fn multiplyValueInplace(self: *Self, v: anytype) void {
    self.calcValueInplace(.multiply, v);
}

pub fn divideValueInplace(self: *Self, v: anytype) void {
    self.calcValueInplace(.divide, v);
}

pub fn calcMat(self: Self, op: OperationType, m: Self, dest: *Self) void {
    _ = switch (op) {
        .add => c.Mat_Add(self.ptr, m.ptr, dest.*.ptr),
        .subtract => c.Mat_Subtract(self.ptr, m.ptr, dest.*.ptr),
        .multiply => c.Mat_Multiply(self.ptr, m.ptr, dest.*.ptr),
        .divide => c.Mat_Divide(self.ptr, m.ptr, dest.*.ptr),
    };
}

// Add calculates the per-element sum of two arrays or an array and a scalar.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga10ac1bfb180e2cfda1701d06c24fdbd6
//
pub fn addMat(self: Self, m: Self, dest: *Self) void {
    self.calcMat(.add, m, dest);
}

// Subtract calculates the per-element subtraction of two arrays or an array and a scalar.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaa0f00d98b4b5edeaeb7b8333b2de353b
//
pub fn subtractMat(self: Self, m: Self, dest: *Self) void {
    self.calcMat(.subtract, m, dest);
}

// Multiply calculates the per-element scaled product of two arrays.
// Both input arrays must be of the same size and the same type.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga979d898a58d7f61c53003e162e7ad89f
//
pub fn multiplyMat(self: Self, m: Self, dest: *Self) void {
    self.calcMat(.multiply, m, dest);
}

// Divide performs the per-element division
// on two arrays or an array and a scalar.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6db555d30115642fedae0cda05604874
//
pub fn divideMat(self: Self, m: Self, dest: *Self) void {
    self.calcMat(.divide, m, dest);
}

/// MultiplyWithParams calculates the per-element scaled product of two arrays.
/// Both input arrays must be of the same size and the same type.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga979d898a58d7f61c53003e162e7ad89f
///
pub fn multiplyWithParams(self: Self, m: Self, scale: f64, dest: *Self) void {
    c.Mat_MultiplyWithParams(self.ptr, m.ptr, scale, dest.*.ptr);
}

// AbsDiff calculates the per-element absolute difference between two arrays
// or between an array and a scalar.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6fef31bc8c4071cbc114a758a2b79c14
//
pub fn absDiff(self: Self, m: Self, dest: *Self) void {
    c.Mat_AbsDiff(self.ptr, m.ptr, dest.*.ptr);
}

// Eigen calculates eigenvalues and eigenvectors of a symmetric matrix.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga9fa0d58657f60eaa6c71f6fbb40456e3
//
pub fn eigen(self: Self, eigenvalues: *Self, eigenvectors: *Self) bool {
    return c.Mat_Eigen(self.ptr, eigenvalues.*.ptr, eigenvectors.*.ptr);
}

// EigenNonSymmetric calculates eigenvalues and eigenvectors of a non-symmetric matrix (real eigenvalues only).
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaf51987e03cac8d171fbd2b327cf966f6
//
pub fn eigenNonSymmetric(self: Self, eigenvalues: *Self, eigenvectors: *Self) void {
    c.Mat_EigenNonSymmetric(self.ptr, eigenvalues.*.ptr, eigenvectors.*.ptr);
}

// Exp calculates the exponent of every array element.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga3e10108e2162c338f1b848af619f39e5
//
pub fn exp(self: Self, dest: *Self) void {
    c.Mat_Exp(self.ptr, dest.*.ptr);
}

// ExtractChannel extracts a single channel from src (coi is 0-based index).
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gacc6158574aa1f0281878c955bcf35642
//
pub fn extractChannel(self: Self, dest: *Self, coi: i32) void {
    c.Mat_ExtractChannel(self.ptr, dest.*.ptr, coi);
}

// FindNonZero returns the list of locations of non-zero pixels.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaed7df59a3539b4cc0fe5c9c8d7586190
//
pub fn findNonZero(self: Self, idx: *Self) void {
    c.Mat_FindNonZero(self.ptr, idx.*.ptr);
}

// Flip flips a 2D array around horizontal(0), vertical(1), or both axes(-1).
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaca7be533e3dac7feb70fc60635adf441
//
pub fn flip(self: Self, dest: *Self, flipCode: i32) void {
    c.Mat_Flip(self.ptr, dest.*.ptr, flipCode);
}

// Gemm performs generalized matrix multiplication.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gacb6e64071dffe36434e1e7ee79e7cb35
//
pub fn gemm(self: Self, m1: Self, alpha: f64, m2: Self, beta: f64, dest: *Self, flags: GemmFlags) void {
    c.Mat_Gemm(self.ptr, m1.ptr, alpha, m2.ptr, beta, dest.*.ptr, @intFromEnum(flags));
}

// GetOptimalDFTSize returns the optimal Discrete Fourier Transform (DFT) size
// for a given vector size.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6577a2e59968936ae02eb2edde5de299
//
pub fn getOptimalDFTSize(vecsize: i32) i32 {
    return c.Mat_GetOptimalDFTSize(vecsize);
}

/// Hconcat applies horizontal concatenation to given matrices.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaab5ceee39e0580f879df645a872c6bf7
///
pub fn hconcat(src1: Self, src2: Self, dst: *Self) void {
    c.Mat_Hconcat(src1.ptr, src2.ptr, dst.*.ptr);
}

/// Vconcat applies vertical concatenation to given matrices.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaab5ceee39e0580f879df645a872c6bf7
///
pub fn vconcat(src1: Self, src2: Self, dst: *Self) void {
    c.Mat_Vconcat(src1.ptr, src2.ptr, dst.*.ptr);
}

/// Rotate rotates a 2D array in multiples of 90 degrees
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga4ad01c0978b0ce64baa246811deeac24
pub fn rotate(self: Self, dest: *Self, rotation_code: RotateFlag) void {
    c.Mat_Rotate(self.ptr, dest.*.ptr, @intFromEnum(rotation_code));
}

/// IDCT calculates the inverse Discrete Cosine Transform of a 1D or 2D array.
/// idct(src, dst, flags) is equivalent to dct(src, dst, flags | DCT_INVERSE).
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga77b168d84e564c50228b69730a227ef2
///
pub fn idct(self: Self, dest: *Self, flags: DftFlags) void {
    c.Mat_IDCT(self.ptr, dest.*.ptr, @intFromEnum(flags));
}

/// IDFT calculates the inverse Discrete Fourier Transform of a 1D or 2D array.
/// idft(src, dst, flags) is equivalent to dft(src, dst, flags | DFT_INVERSE) .
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaa708aa2d2e57a508f968eb0f69aa5ff1
///
pub fn idft(self: Self, dest: *Self, flags: DftFlags, nonzeroRows: i32) void {
    c.Mat_IDFT(self.ptr, dest.*.ptr, @intFromEnum(flags), nonzeroRows);
}

/// InRange checks if array elements lie between the elements of two Mat arrays.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981
///
pub fn inRange(self: Self, lowerb: Self, upperb: Self, dest: *Self) void {
    c.Mat_InRange(self.ptr, lowerb.ptr, upperb.ptr, dest.*.ptr);
}

/// InRangeWithScalar checks if array elements lie between the elements of two Scalars
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981
///
pub fn inRangeWithScalar(self: Self, lowerb: Scalar, upperb: Scalar, dest: *Self) void {
    c.Mat_InRangeWithScalar(self.ptr, lowerb, upperb, dest.*.ptr);
}

/// InsertChannel inserts a single channel to dst (coi is 0-based index)
/// (it replaces channel i with another in dst).
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga1d4bd886d35b00ec0b764cb4ce6eb515
///
//     pub extern fn Mat_InsertChannel(src: Mat, dst: Mat, coi: c_int) void;
pub fn insertChannel(self: Self, dest: *Self, coi: i32) void {
    c.Mat_InsertChannel(self.ptr, dest.*.ptr, coi);
}

/// Invert finds the inverse or pseudo-inverse of a matrix.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#gad278044679d4ecf20f7622cc151aaaa2
///
//     pub extern fn Mat_Invert(src: Mat, dst: Mat, flags: c_int) f64;
pub fn invert(self: Self, dest: *Self, flags: SolveDecompositionFlag) f64 {
    return c.Mat_Invert(self.ptr, dest.*.ptr, @intFromEnum(flags));
}

/// KMeans finds centers of clusters and groups input samples around the clusters.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88
///
pub fn kmeans(data: Self, k: i32, bestLabels: *Self, criteria: TermCriteria, attempts: i32, flags: KMeansFlag, centers: *Self) f64 {
    return c.KMeans(data.ptr, k, bestLabels.*.ptr, criteria, attempts, @intFromEnum(flags), centers.*.ptr);
}

/// KMeansPoints finds centers of clusters and groups input samples around the clusters.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88
///
pub fn kmeansPoints(pts: PointVector, k: i32, bestLabels: *Self, criteria: TermCriteria, attempts: i32, flags: KMeansFlag, centers: *Self) f64 {
    return c.KMeansPoints(pts.ptr, k, bestLabels.*.ptr, criteria, attempts, @intFromEnum(flags), centers.*.ptr);
}

/// Log calculates the natural logarithm of every array element.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga937ecdce4679a77168730830a955bea7
///
//     pub extern fn Mat_Log(src: Mat, dst: Mat) void;
pub fn log(self: Self, dest: *Self) void {
    c.Mat_Log(self.ptr, dest.*.ptr);
}

/// Magnitude calculates the magnitude of 2D vectors.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6d3b097586bca4409873d64a90fe64c3
///
pub fn magnitude(x: Self, y: Self, magnitude_: *Self) void {
    c.Mat_Magnitude(x.ptr, y.ptr, magnitude_.*.ptr);
}

/// Max calculates per-element maximum of two arrays or an array and a scalar.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#gacc40fa15eac0fb83f8ca70b7cc0b588d
///
pub fn max(self: Self, other: Self, dest: *Self) void {
    c.Mat_Max(self.ptr, other.ptr, dest.*.ptr);
}

/// MeanStdDev calculates a mean and standard deviation of array elements.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga846c858f4004d59493d7c6a4354b301d
///
pub fn meanStdDev(self: Self, dst_mean: *Self, dst_std_dev: *Self) void {
    c.Mat_MeanStdDev(self.ptr, dst_mean.*.ptr, dst_std_dev.*.ptr);
}

/// Merge creates one multi-channel array out of several single-channel ones.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga7d7b4d6c6ee504b30a20b1680029c7b4
///
pub fn merge(mats: []Self, dest: *Self) void {
    const c_mats = toCStructs(mats);
    defer deinitCStructs(c_mats);
    c.Mat_Merge(c_mats, dest.*.ptr);
}

/// Min calculates per-element minimum of two arrays or an array and a scalar.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga9af368f182ee76d0463d0d8d5330b764
///
pub fn min(self: Self, other: Self, dest: *Self) void {
    c.Mat_Min(self.ptr, other.ptr, dest.*.ptr);
}

/// AddWeighted calculates the weighted sum of two arrays.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19
///
pub fn addMatWeighted(self: Self, alpha: f64, m: Self, beta: f64) Self {
    const dest = self.init();
    _ = c.Mat_AddWeighted(self.ptr, alpha, m.ptr, beta, dest.ptr);
    return dest;
}

pub fn bitwise(self: Self, m: Self, dest: *Self, comptime op: BitOperationType) void {
    _ = switch (op) {
        .and_ => _ = c.Mat_BitwiseAnd(self.ptr, m.ptr, dest.*.ptr),
        .not_ => _ = c.Mat_BitwiseNot(self.ptr, dest.*.ptr),
        .or_ => _ = c.Mat_BitwiseOr(self.ptr, m.ptr, dest.*.ptr),
        .xor_ => _ = c.Mat_BitwiseXor(self.ptr, m.ptr, dest.*.ptr),
    };
}

// BitwiseAnd computes bitwise conjunction of the two arrays (dst = src1 & src2).
// Calculates the per-element bit-wise conjunction of two arrays
// or an array and a scalar.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga60b4d04b251ba5eb1392c34425497e14
//
pub fn bitwiseAnd(self: Self, m: Self, dest: *Self) void {
    return self.bitwise(m, dest, .and_);
}

// BitwiseNot inverts every bit of an array.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga0002cf8b418479f4cb49a75442baee2f
//
pub fn bitwiseNot(self: Self, dest: *Self) void {
    return self.bitwise(self, dest, .not_);
}

// BitwiseOr calculates the per-element bit-wise disjunction of two arrays
// or an array and a scalar.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gab85523db362a4e26ff0c703793a719b4
//

pub fn bitwiseOr(self: Self, m: Self, dest: *Self) void {
    return self.bitwise(m, dest, .or_);
}

// BitwiseXor calculates the per-element bit-wise "exclusive or" operation
// on two arrays or an array and a scalar.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga84b2d8188ce506593dcc3f8cd00e8e2c
//
pub fn bitwiseXor(self: Self, m: Self, dest: *Self) void {
    return self.bitwise(m, dest, .xor_);
}

pub fn bitwiseWithMask(self: Self, comptime op: BitOperationType, m: Self, dest: *Self, mask: Self) void {
    _ = switch (op) {
        .and_ => _ = c.Mat_BitwiseAndWithMask(self.ptr, m.ptr, dest.*.ptr, mask.ptr),
        .not_ => _ = c.Mat_BitwiseNotWithMask(self.ptr, dest.*.ptr, mask.ptr),
        .or_ => _ = c.Mat_BitwiseOrWithMask(self.ptr, m.ptr, dest.*.ptr, mask.ptr),
        .xor_ => _ = c.Mat_BitwiseXorWithMask(self.ptr, m.ptr, dest.*.ptr, mask.ptr),
    };
}

// BitwiseAndWithMask computes bitwise conjunction of the two arrays (dst = src1 & src2).
// Calculates the per-element bit-wise conjunction of two arrays
// or an array and a scalar. It has an additional parameter for a mask.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga60b4d04b251ba5eb1392c34425497e14
//
pub fn bitwiseAndWithMask(self: Self, m: Self, dest: *Self, mask: Self) void {
    return self.bitwiseWithMask(self, .and_, m, dest, mask);
}

// BitwiseNotWithMask inverts every bit of an array. It has an additional parameter for a mask.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga0002cf8b418479f4cb49a75442baee2f
//
pub fn bitwiseNotWithMask(self: Self, m: Self, dest: *Self, mask: Self) void {
    return self.bitwiseWithMask(self, .not_, m, dest, mask);
}

// BitwiseOrWithMask calculates the per-element bit-wise disjunction of two arrays
// or an array and a scalar. It has an additional parameter for a mask.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gab85523db362a4e26ff0c703793a719b4
//
pub fn bitwiseOrWithMask(self: Self, m: Self, dest: *Self, mask: Self) void {
    return self.bitwiseWithMask(self, .or_, m, dest, mask);
}

// BitwiseXorWithMask calculates the per-element bit-wise "exclusive or" operation
// on two arrays or an array and a scalar. It has an additional parameter for a mask.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga84b2d8188ce506593dcc3f8cd00e8e2c
//
pub fn bitwiseXorWithMask(self: Self, m: Self, dest: *Self, mask: Self) void {
    return self.bitwiseWithMask(self, .xor_, m, dest, mask);
}

pub fn compare(self: Self, m: Self, dest: *Self, comptime op: CompareType) void {
    _ = c.Mat_Compare(self.ptr, m.ptr, dest.*.ptr, @intFromEnum(op));
}

// BatchDistance is a naive nearest neighbor finder.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga4ba778a1c57f83233b1d851c83f5a622
//
pub fn batchDistance(self: Self, m: Self, dest: *Self, dtype: c_int, nidx: *Self, normType: c_int, K: c_int, mask: *Self, update: c_int, crosscheck: bool) void {
    _ = c.Mat_BatchDistance(self.ptr, m.ptr, dest.*.ptr, dtype, nidx.*.ptr, normType, K, mask.*.ptr, update, crosscheck);
}

// BorderInterpolate computes the source location of an extrapolated pixel.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga247f571aa6244827d3d798f13892da58
//
pub fn borderInterpolate(p: c_int, len: c_int, border_type: BorderType) c_int {
    return c.Mat_BorderInterpolate(p, len, border_type.toNum());
}

// CalcCovarMatrix calculates the covariance matrix of a set of vectors.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga017122d912af19d7d0d2cccc2d63819f
//
pub fn calcCovarMatrix(samples: Self, covar: *Self, mean_: *Self, flags: CovarFlags, ctype: MatType) void {
    _ = c.Mat_CalcCovarMatrix(samples.ptr, covar.*.ptr, mean_.*.ptr, flags.toNum(), @intFromEnum(ctype));
}

// CartToPolar calculates the magnitude and angle of 2D vectors.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gac5f92f48ec32cacf5275969c33ee837d
//
pub fn cartToPolar(x: Self, y: Self, magnitude_: *Self, angle: *Self, angle_in_degrees: bool) void {
    _ = c.Mat_CartToPolar(x.ptr, y.ptr, magnitude_.*.ptr, angle.*.ptr, angle_in_degrees);
}

// CheckRange checks every element of an input array for invalid values.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga2bd19d89cae59361416736f87e3c7a64
//
pub fn checkRange(self: Self) bool {
    return c.Mat_CheckRange(self.ptr);
}

// CountNonZero counts non-zero array elements.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaa4b89393263bb4d604e0fe5986723914
//
pub fn countNonZero(self: Self) i32 {
    return c.Mat_CountNonZero(self.ptr);
}

/// DCT performs a forward or inverse discrete Cosine transform of 1D or 2D array.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga85aad4d668c01fbd64825f589e3696d4
///
pub fn dct(self: Self, dst: *Self, flags: DftFlags) void {
    c.Mat_DCT(self.ptr, dst.*.ptr, flags.toNum());
}

/// Determinant returns the determinant of a square floating-point matrix.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaf802bd9ca3e07b8b6170645ef0611d0c
///
pub fn determinant(self: Self) f64 {
    return c.Mat_Determinant(self.ptr);
}

/// DFT performs a forward or inverse Discrete Fourier Transform (DFT)
/// of a 1D or 2D floating-point array.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#gadd6cf9baf2b8b704a11b5f04aaf4f39d
///
pub fn dft(self: Self, dst: *Self, flags: DftFlags) void {
    c.Mat_DFT(self.ptr, dst.*.ptr, flags.toNum());
}

// CompleteSymm copies the lower or the upper half of a square matrix to its another half.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaa9d88dcd0e54b6d1af38d41f2a3e3d25
//
pub fn completeSymm(self: Self, lower_to_upper: bool) void {
    return c.Mat_CompleteSymm(self.ptr, lower_to_upper);
}

// ConvertScaleAbs scales, calculates absolute values, and converts the result to 8-bit.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga3460e9c9f37b563ab9dd550c4d8c4e7d
//
pub fn convertScaleAbs(self: Self, dest: *Self, alpha: f64, beta: f64) void {
    return c.Mat_ConvertScaleAbs(self.ptr, dest.*.ptr, alpha, beta);
}

// CopyMakeBorder forms a border around an image (applies padding).
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga2ac1049c2c3dd25c2b41bffe17658a36
//
pub fn copyMakeBorder(self: Self, dest: *Self, top: i32, bottom: i32, left: i32, right: i32, borderType: BorderType, value: Color) void {
    return c.Mat_CopyMakeBorder(self.ptr, dest.*.ptr, top, bottom, left, right, borderType.toNum(), value.toScalar().toC());
}

pub fn dataPtr(self: Self, comptime T: type) ![]T {
    if (switch (T) {
        u8 => @intFromEnum(self.getType()) & MatType.cv8u != MatType.cv8u,
        i8 => @intFromEnum(self.getType()) & MatType.cv8s != MatType.cv8s,
        u16 => @intFromEnum(self.getType()) & MatType.cv16u != MatType.cv16u,
        i16 => @intFromEnum(self.getType()) & MatType.cv16s != MatType.cv16s,
        f32 => @intFromEnum(self.getType()) & MatType.cv32f != MatType.cv32f,
        f64 => @intFromEnum(self.getType()) & MatType.cv64f != MatType.cv64f,
        else => @compileError("Unsupported type"),
    }) {
        return error.RuntimeError;
    }

    if (!self.isContinuous()) {
        return error.RuntimeError;
    }

    const p: c.ByteArray = c.Mat_DataPtr(self.ptr);
    const len = @as(usize, @intCast(p.length));
    const bit_scale = @sizeOf(T) / @sizeOf(u8);
    return @as([*]T, @ptrCast(@alignCast(p.data)))[0 .. len / bit_scale];
}

pub fn toBytes(self: Self) []u8 {
    var p: c.struct_ByteArray = c.Mat_ToBytes(self.ptr);
    const len = @as(usize, @intCast(p.length));
    return p[0..len];
}

// Reshape changes the shape and/or the number of channels of a 2D matrix without copying the data.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#a4eb96e3251417fa88b78e2abd6cfd7d8
//
pub fn reshape(self: Self, cn: i32, rows_: usize) !Self {
    const ptr = c.Mat_Reshape(self.ptr, @as(u31, @intCast(cn)), @as(i32, @intCast(rows_)));
    return try initFromC(ptr);
}

// Region returns a new Mat that points to a region of this Mat. Changes made to the
// region Mat will affect the original Mat, since they are pointers to the underlying
// OpenCV Mat object.
pub fn region(self: Self, r: Rect) Self {
    const ptr = c.Mat_Region(self.ptr, r.toC());
    return initFromC(ptr);
}

// T  transpose matrix
// https://docs.opencv.org/4.1.2/d3/d63/classcv_1_1Mat.html#aaa428c60ccb6d8ea5de18f63dfac8e11
pub fn t(self: Self) Self {
    const ptr = c.Mat_T(self.ptr);
    return initFromC(ptr);
}

// Transpose transposes a matrix.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga46630ed6c0ea6254a35f447289bd7404
//
pub fn transpose(self: Self, dst: *Self) Self {
    _ = c.Mat_Transpose(self.ptr, dst.*.ptr);
}

// PolatToCart calculates x and y coordinates of 2D vectors from their magnitude and angle.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga581ff9d44201de2dd1b40a50db93d665
//
pub fn polarToCart(magnitude_: Self, degree: Self, x: *Self, y: *Self, angleInDegrees: bool) void {
    _ = c.Mat_PolarToCart(magnitude_.ptr, degree.ptr, x.*.ptr, y.*.ptr, angleInDegrees);
}

// Pow raises every array element to a power.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaf0d056b5bd1dc92500d6f6cf6bac41ef
//
pub fn pow(self: Self, power: f64, dst: *Self) void {
    _ = c.Mat_Pow(self.ptr, power, dst.*.ptr);
}

// Phase calculates the rotation angle of 2D vectors.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga9db9ca9b4d81c3bde5677b8f64dc0137
//
pub fn phase(x: Self, y: Self, angle: *Self, angle_in_degrees: bool) void {
    _ = c.Mat_Phase(x.ptr, y.ptr, angle.*.ptr, angle_in_degrees);
}

// LUT performs a look-up table transform of an array.
//
// The function LUT fills the output array with values from the look-up table.
// Indices of the entries are taken from the input array.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gab55b8d062b7f5587720ede032d34156f
pub fn lut(src: Self, lut_: Self, dst: *Self) void {
    c.LUT(src.ptr, lut_.ptr, dst.*.ptr);
}

// Sum calculates the per-channel pixel sum of an image.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga716e10a2dd9e228e4d3c95818f106722
//
//     pub extern fn Mat_Sum(src1: Mat) Scalar;
pub fn sum(self: Self) Scalar {
    return Scalar.initFromC(c.Mat_Sum(self.ptr));
}

// RowRange creates a matrix header for the specified row span.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#aa6542193430356ad631a9beabc624107
//
pub fn rowRange(self: Self, startrow: i32, endrow: i32) Self {
    const ptr = c.Mat_rowRange(self.ptr, startrow, endrow);
    return initFromC(ptr);
}

// ColRange creates a matrix header for the specified column span.
//
// For further details, please see:
// https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html#aadc8f9210fe4dec50513746c246fa8d9
//
pub fn colRange(self: Self, startcol: i32, endcol: i32) Self {
    const ptr = c.Mat_colRange(self.ptr, startcol, endcol);
    return initFromC(ptr);
}

// PatchNaNs converts NaN's to zeros.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga62286befb7cde3568ff8c7d14d5079da
//
pub fn patchNaNs(self: Self) void {
    c.Mat_PatchNaNs(self.ptr);
}

// RandN Fills the array with normally distributed random numbers.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gaeff1f61e972d133a04ce3a5f81cf6808
//
pub fn randN(self: *Self, mean_: Scalar, stddev: Scalar) void {
    _ = c.RandN(self.ptr, mean_.toC(), stddev.toC());
}

// RandShuffle Shuffles the array elements randomly.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6a789c8a5cb56c6dd62506179808f763
//
pub fn randShuffle(self: *Self) void {
    _ = c.RandShuffle(self.ptr);
}

// RandShuffleWithParams Shuffles the array elements randomly.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6a789c8a5cb56c6dd62506179808f763
//
pub fn randShuffleWithParams(self: *Self, iter_factor: f64, rng: RNG) void {
    _ = c.RandShuffleWithParams(self.ptr, iter_factor, rng.toC());
}

// RandU Generates a single uniformly-distributed random
// number or an array of random numbers.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga1ba1026dca0807b27057ba6a49d258c0
//
pub fn randU(self: *Self, low: Scalar, high: Scalar) void {
    _ = c.RandU(self.ptr, low.toC(), high.toC());
}

// Solve solves one or more linear systems or least-squares problems.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga12b43690dbd31fed96f213eefead2373
//
pub fn solve(self: Self, src2: Self, dst: *Self, flag: SolveDecompositionFlag) bool {
    return c.Mat_Solve(self.ptr, src2.ptr, dst.*.ptr, @intFromEnum(flag));
}
// SolveCubic finds the real roots of a cubic equation.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga1c3b0b925b085b6e96931ee309e6a1da
//
pub fn solveCubic(self: Self, roots: *Self) bool {
    return c.Mat_SolveCubic(self.ptr, roots.*.ptr);
}

/// Normalize normalizes the norm or value range of an array.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga87eef7ee3970f86906d69a92cbf064bd
///
pub fn normalize(self: Self, dst: *Self, alpha: f64, beta: f64, typ: NormType) void {
    c.Mat_Normalize(self.ptr, dst.*.ptr, alpha, beta, @intFromEnum(typ));
}

/// Norm calculates the absolute norm of an array.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga7c331fb8dd951707e184ef4e3f21dd33
///
pub fn norm(self: Self, normType: NormType) f64 {
    return c.Norm(self.ptr, @intFromEnum(normType));
}

/// Norm calculates the absolute difference/relative norm of two arrays.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga7c331fb8dd951707e184ef4e3f21dd33
///
pub fn normWithMats(self: Self, src2: Self, norm_type: NormType) f64 {
    return c.NormWithMats(self.ptr, src2.ptr, @intFromEnum(norm_type));
}

/// PerspectiveTransform performs the perspective matrix transformation of vectors.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7
///
pub fn perspectiveTransform(self: Self, dst: *Self, m: Self) void {
    c.Mat_PerspectiveTransform(self.ptr, dst.*.ptr, m.ptr);
}

// SolvePoly finds the real or complex roots of a polynomial equation.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#gac2f5e953016fabcdf793d762f4ec5dce
//
pub fn solvePoly(self: Self, roots: *Self, max_iters: i32) bool {
    return c.Mat_SolvePoly(self.ptr, roots.*.ptr, max_iters);
}

pub fn reduce(self: Self, dst: *Self, dim: i32, r_type: ReduceType, d_type: MatType) void {
    c.Mat_Reduce(self.ptr, dst.*.ptr, dim, @intFromEnum(r_type), @intFromEnum(d_type));
}

/// Repeat fills the output array with repeated copies of the input array.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga496c3860f3ac44c40b48811333cfda2d
///
pub fn repeat(self: Self, ny: i32, nx: i32, dst: *Self) void {
    c.Mat_Repeat(self.ptr, ny, nx, dst.*.ptr);
}

/// Calculates the sum of a scaled array and another array.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga9e0845db4135f55dcf20227402f00d98
///
pub fn scaleAdd(self: Self, alpha: f64, src2: Self, dst: *Self) void {
    c.Mat_ScaleAdd(self.ptr, alpha, src2.ptr, dst.*.ptr);
}

/// SetIdentity initializes a scaled identity matrix.
/// For further details, please see:
///  https://docs.opencv.org/master/d2/de8/group__core__array.html#ga388d7575224a4a277ceb98ccaa327c99
///
pub fn setIdentity(self: Self, s: f64) void {
    c.Mat_SetIdentity(self.ptr, s);
}

/// Sort sorts each row or each column of a matrix.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga45dd56da289494ce874be2324856898f
///
pub fn sort(self: Self, dst: *Self, flags: SortFlags) void {
    c.Mat_Sort(self.ptr, dst.*.ptr, @intFromEnum(flags));
}

/// SortIdx sorts each row or each column of a matrix.
/// Instead of reordering the elements themselves, it stores the indices of sorted elements in the output array
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#gadf35157cbf97f3cb85a545380e383506
///
pub fn sortIdx(self: Self, dst: *Self, flags: SortFlags) void {
    c.Mat_SortIdx(self.ptr, dst.*.ptr, @intFromEnum(flags));
}

// Split creates an array of single channel images from a multi-channel image
// Created images should be closed manualy to avoid memory leaks.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga0547c7fed86152d7e9d0096029c8518a
//
pub fn split(self: Self, allocator: std.mem.Allocator) !Mats {
    var c_mats: c.struct_Mats = undefined;
    c.Mat_Split(self.ptr, &c_mats);
    const mats = try toArrayList(c_mats, allocator);
    return .{ .list = mats };
}

/// Trace returns the trace of a matrix.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga3419ac19c7dcd2be4bd552a23e147dd8
///
pub fn trace(self: Self) Scalar {
    const s = c.Mat_Trace(self.ptr);
    return Scalar.initFromC(s);
}

/// Transform performs the matrix transformation of every array element.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga393164aa54bb9169ce0a8cc44e08ff22
///
pub fn transform(src: Self, dst: *Self, tm: Self) void {
    c.Mat_Transform(src.ptr, dst.*.ptr, tm.ptr);
}

/// MinMaxIdx finds the global minimum and maximum in an array.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga7622c466c628a75d9ed008b42250a73f
///
pub fn minMaxIdx(self: Self) struct {
    min_val: f64,
    max_val: f64,
    min_idx: i32,
    max_idx: i32,
} {
    var min_val: f64 = undefined;
    var max_val: f64 = undefined;
    var min_idx: i32 = undefined;
    var max_idx: i32 = undefined;
    c.Mat_MinMaxIdx(
        self.ptr,
        &min_val,
        &max_val,
        &min_idx,
        &max_idx,
    );
    return .{
        .min_val = min_val,
        .max_val = max_val,
        .min_idx = min_idx,
        .max_idx = max_idx,
    };
}

/// MinMaxLoc finds the global minimum and maximum in an array.
///
/// For further details, please see:
/// https://docs.opencv.org/trunk/d2/de8/group__core__array.html#gab473bf2eb6d14ff97e89b355dac20707
///
pub fn minMaxLoc(self: Self) struct {
    min_val: f64,
    max_val: f64,
    min_loc: Point,
    max_loc: Point,
} {
    var min_val: f64 = undefined;
    var max_val: f64 = undefined;
    var c_min_loc: c.Point = undefined;
    var c_max_loc: c.Point = undefined;

    c.Mat_MinMaxLoc(
        self.ptr,
        &min_val,
        &max_val,
        &c_min_loc,
        &c_max_loc,
    );

    const min_loc = Point.initFromC(c_min_loc);
    const max_loc = Point.initFromC(c_max_loc);

    return .{
        .min_val = min_val,
        .max_val = max_val,
        .min_loc = min_loc,
        .max_loc = max_loc,
    };
}

/// Copies specified channels from input arrays to the specified channels of output arrays.
///
/// For further details, please see:
/// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga51d768c270a1cdd3497255017c4504be
///
//     pub extern fn Mat_MixChannels(src: struct_Mats, dst: struct_Mats, fromTo: struct_IntVector) void;
pub fn mixChannels(src: []Self, dst: *[]Self, from_to: []i32) !void {
    const c_src = toCStructs(src);
    defer deinitCStructs(c_src);
    const c_dst = toCStructs(dst.*);
    defer deinitCStructs(c_dst);
    const c_from_to = c.struct_IntVector{
        .val = @as([*]i32, @ptrCast(from_to.ptr)),
        .length = @as(i32, @intCast(from_to.len)),
    };
    c.Mat_MixChannels(c_src, c_dst, c_from_to);

    for (dst.*, 0..) |*d, i| {
        d.*.deinit();
        d.* = try Self.initFromC(c_dst.val[i]);
    }
}

//Mulspectrums performs the per-element multiplication of two Fourier spectrums.
//
// For further details, please see:
// https://docs.opencv.org/master/d2/de8/group__core__array.html#ga3ab38646463c59bf0ce962a9d51db64f
//
pub fn mulSpectrums(self: Self, src2: Self, dst: *Self, flags: DftFlags) void {
    c.Mat_MulSpectrums(self.ptr, src2.ptr, dst.*.ptr, @intFromEnum(flags));
}

pub fn toArrayList(c_mats: c.Mats, allocator: std.mem.Allocator) !Mats {
    const mat_array = try utils.fromCStructsToArrayList(c_mats.mats, c_mats.length, Self, allocator);
    return .{ .list = mat_array };
}

pub inline fn toCStructs(mats: []const Self) !c.Mats {
    const len = @as(i32, @intCast(mats.len));
    var c_mats = c.Mats_New(len);
    if (c_mats.length != len) return error.AllocationError;
    _ = try epnn(c_mats.mats);
    {
        var i: usize = 0;
        while (i < mats.len) : (i += 1) {
            c_mats.mats[i] = mats[i].toC();
        }
    }
    return c_mats;
}

pub fn deinitCStructs(c_mats: c.Mats) void {
    c.Mats_Close(c_mats);
}

pub const Mats = struct {
    list: std.ArrayList(Self),

    pub fn deinit(self: *Mats) void {
        for (self.list.items) |*m| {
            m.deinit();
        }
        self.list.deinit();
    }
};

test "core mat" {
    _ = @import("mat_test.zig");
    _ = DftFlags;
    _ = CovarFlags;
    _ = SortFlags;
}

//*    implementation done ("i" is internal function so we don't write zig wrappers for them)
//i    pub extern fn Mats_get(mats: struct_Mats, i: c_int) Mat;
//i    pub extern fn Mats_Close(mats: struct_Mats) void;
//*    pub extern fn Mat_New(...) Mat;
//*    pub extern fn Mat_NewWithSize(rows: c_int, cols: c_int, @"type": c_int) Mat;
//*    pub extern fn Mat_NewWithSizes(sizes: struct_IntVector, @"type": c_int) Mat;
//*    pub extern fn Mat_NewWithSizesFromScalar(sizes: IntVector, @"type": c_int, ar: Scalar) Mat;
//*    pub extern fn Mat_NewWithSizesFromBytes(sizes: IntVector, @"type": c_int, buf: struct_ByteArray) Mat;
//*    pub extern fn Mat_NewFromScalar(ar: Scalar, @"type": c_int) Mat;
//*    pub extern fn Mat_NewWithSizeFromScalar(ar: Scalar, rows: c_int, cols: c_int, @"type": c_int) Mat;
//*    pub extern fn Mat_NewFromBytes(rows: c_int, cols: c_int, @"type": c_int, buf: struct_ByteArray) Mat;
//*    pub extern fn Mat_FromPtr(m: Mat, rows: c_int, cols: c_int, @"type": c_int, prows: c_int, pcols: c_int) Mat;
//*    pub extern fn Mat_Close(m: Mat) void;
//*    pub extern fn Mat_Empty(m: Mat) c_int;
//*    pub extern fn Mat_IsContinuous(m: Mat) bool;
//*    pub extern fn Mat_Clone(m: Mat) Mat;
//*    pub extern fn Mat_CopyTo(m: Mat, dst: Mat) void;
//*    pub extern fn Mat_Total(m: Mat) c_int;
//*    pub extern fn Mat_Size(m: Mat, res: [*c]IntVector) void;
//*    pub extern fn Mat_CopyToWithMask(m: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_ConvertTo(m: Mat, dst: Mat, @"type": c_int) void;
//*    pub extern fn Mat_ConvertToWithParams(m: Mat, dst: Mat, @"type": c_int, alpha: f32, beta: f32) void;
//*    pub extern fn Mat_ToBytes(m: Mat) struct_ByteArray;
//*    pub extern fn Mat_DataPtr(m: Mat) struct_ByteArray;
//*    pub extern fn Mat_Region(m: Mat, r: Rect) Mat;
//*    pub extern fn Mat_Reshape(m: Mat, cn: c_int, rows: c_int) Mat;
//*    pub extern fn Mat_PatchNaNs(m: Mat) void;
//*    pub extern fn Mat_ConvertFp16(m: Mat) Mat;
//*    pub extern fn Mat_Mean(m: Mat) Scalar;
//*    pub extern fn Mat_MeanWithMask(m: Mat, mask: Mat) Scalar;
//*    pub extern fn Mat_Sqrt(m: Mat) Mat;
//*    pub extern fn Mat_Rows(m: Mat) c_int;
//*    pub extern fn Mat_Cols(m: Mat) c_int;
//*    pub extern fn Mat_Channels(m: Mat) c_int;
//*    pub extern fn Mat_Type(m: Mat) c_int;
//*    pub extern fn Mat_Step(m: Mat) c_int;
//*    pub extern fn Mat_ElemSize(m: Mat) c_int;
//*    pub extern fn Eye(rows: c_int, cols: c_int, @"type": c_int) Mat;
//*    pub extern fn Zeros(rows: c_int, cols: c_int, @"type": c_int) Mat;
//*    pub extern fn Ones(rows: c_int, cols: c_int, @"type": c_int) Mat;
//*    pub extern fn Mat_GetUChar(m: Mat, row: c_int, col: c_int) u8;
//*    pub extern fn Mat_GetUChar3(m: Mat, x: c_int, y: c_int, z: c_int) u8;
//*    pub extern fn Mat_GetSChar(m: Mat, row: c_int, col: c_int) i8;
//*    pub extern fn Mat_GetSChar3(m: Mat, x: c_int, y: c_int, z: c_int) i8;
//*    pub extern fn Mat_GetShort(m: Mat, row: c_int, col: c_int) i16;
//*    pub extern fn Mat_GetShort3(m: Mat, x: c_int, y: c_int, z: c_int) i16;
//*    pub extern fn Mat_GetInt(m: Mat, row: c_int, col: c_int) i32;
//*    pub extern fn Mat_GetInt3(m: Mat, x: c_int, y: c_int, z: c_int) i32;
//*    pub extern fn Mat_GetFloat(m: Mat, row: c_int, col: c_int) f32;
//*    pub extern fn Mat_GetFloat3(m: Mat, x: c_int, y: c_int, z: c_int) f32;
//*    pub extern fn Mat_GetDouble(m: Mat, row: c_int, col: c_int) f64;
//*    pub extern fn Mat_GetDouble3(m: Mat, x: c_int, y: c_int, z: c_int) f64;
//*    pub extern fn Mat_SetTo(m: Mat, value: Scalar) void;
//*    pub extern fn Mat_SetUChar(m: Mat, row: c_int, col: c_int, val: u8) void;
//*    pub extern fn Mat_SetUChar3(m: Mat, x: c_int, y: c_int, z: c_int, val: u8) void;
//*    pub extern fn Mat_SetSChar(m: Mat, row: c_int, col: c_int, val: i8) void;
//*    pub extern fn Mat_SetSChar3(m: Mat, x: c_int, y: c_int, z: c_int, val: i8) void;
//*    pub extern fn Mat_SetShort(m: Mat, row: c_int, col: c_int, val: i16) void;
//*    pub extern fn Mat_SetShort3(m: Mat, x: c_int, y: c_int, z: c_int, val: i16) void;
//*    pub extern fn Mat_SetInt(m: Mat, row: c_int, col: c_int, val: i32) void;
//*    pub extern fn Mat_SetInt3(m: Mat, x: c_int, y: c_int, z: c_int, val: i32) void;
//*    pub extern fn Mat_SetFloat(m: Mat, row: c_int, col: c_int, val: f32) void;
//*    pub extern fn Mat_SetFloat3(m: Mat, x: c_int, y: c_int, z: c_int, val: f32) void;
//*    pub extern fn Mat_SetDouble(m: Mat, row: c_int, col: c_int, val: f64) void;
//*    pub extern fn Mat_SetDouble3(m: Mat, x: c_int, y: c_int, z: c_int, val: f64) void;
//*    pub extern fn Mat_AddUChar(m: Mat, val: u8) void;
//*    pub extern fn Mat_SubtractUChar(m: Mat, val: u8) void;
//*    pub extern fn Mat_MultiplyUChar(m: Mat, val: u8) void;
//*    pub extern fn Mat_DivideUChar(m: Mat, val: u8) void;
//*    pub extern fn Mat_AddFloat(m: Mat, val: f32) void;
//*    pub extern fn Mat_SubtractFloat(m: Mat, val: f32) void;
//*    pub extern fn Mat_MultiplyFloat(m: Mat, val: f32) void;
//*    pub extern fn Mat_DivideFloat(m: Mat, val: f32) void;
//*    pub extern fn Mat_MultiplyMatrix(x: Mat, y: Mat) Mat;
//*    pub extern fn Mat_T(x: Mat) Mat;
//*    pub extern fn LUT(src: Mat, lut: Mat, dst: Mat) void;
//*    pub extern fn Mat_AbsDiff(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_Add(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_AddWeighted(src1: Mat, alpha: f64, src2: Mat, beta: f64, gamma: f64, dst: Mat) void;
//*    pub extern fn Mat_BitwiseAnd(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_BitwiseAndWithMask(src1: Mat, src2: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_BitwiseNot(src1: Mat, dst: Mat) void;
//*    pub extern fn Mat_BitwiseNotWithMask(src1: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_BitwiseOr(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_BitwiseOrWithMask(src1: Mat, src2: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_BitwiseXor(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_BitwiseXorWithMask(src1: Mat, src2: Mat, dst: Mat, mask: Mat) void;
//*    pub extern fn Mat_Compare(src1: Mat, src2: Mat, dst: Mat, ct: c_int) void;
//*    pub extern fn Mat_BatchDistance(src1: Mat, src2: Mat, dist: Mat, dtype: c_int, nidx: Mat, normType: c_int, K: c_int, mask: Mat, update: c_int, crosscheck: bool) void;
//*    pub extern fn Mat_BorderInterpolate(p: c_int, len: c_int, borderType: c_int) c_int;
//*    pub extern fn Mat_CalcCovarMatrix(samples: Mat, covar: Mat, mean: Mat, flags: c_int, ctype: c_int) void;
//*    pub extern fn Mat_CartToPolar(x: Mat, y: Mat, magnitude: Mat, angle: Mat, angleInDegrees: bool) void;
//*    pub extern fn Mat_CheckRange(m: Mat) bool;
//*    pub extern fn Mat_CompleteSymm(m: Mat, lowerToUpper: bool) void;
//*    pub extern fn Mat_ConvertScaleAbs(src: Mat, dst: Mat, alpha: f64, beta: f64) void;
//*    pub extern fn Mat_CopyMakeBorder(src: Mat, dst: Mat, top: c_int, bottom: c_int, left: c_int, right: c_int, borderType: c_int, value: Scalar) void;
//*    pub extern fn Mat_CountNonZero(src: Mat) c_int;
//*    pub extern fn Mat_DCT(src: Mat, dst: Mat, flags: c_int) void;
//*    pub extern fn Mat_Determinant(m: Mat) f64;
//*    pub extern fn Mat_DFT(m: Mat, dst: Mat, flags: c_int) void;
//*    pub extern fn Mat_Divide(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_Eigen(src: Mat, eigenvalues: Mat, eigenvectors: Mat) bool;
//*    pub extern fn Mat_EigenNonSymmetric(src: Mat, eigenvalues: Mat, eigenvectors: Mat) void;
//*    pub extern fn Mat_Exp(src: Mat, dst: Mat) void;
//*    pub extern fn Mat_ExtractChannel(src: Mat, dst: Mat, coi: c_int) void;
//*    pub extern fn Mat_FindNonZero(src: Mat, idx: Mat) void;
//*    pub extern fn Mat_Flip(src: Mat, dst: Mat, flipCode: c_int) void;
//*    pub extern fn Mat_Gemm(src1: Mat, src2: Mat, alpha: f64, src3: Mat, beta: f64, dst: Mat, flags: c_int) void;
//*    pub extern fn Mat_GetOptimalDFTSize(vecsize: c_int) c_int;
//*    pub extern fn Mat_Hconcat(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_Vconcat(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Rotate(src: Mat, dst: Mat, rotationCode: c_int) void;
//*    pub extern fn Mat_Idct(src: Mat, dst: Mat, flags: c_int) void;
//*    pub extern fn Mat_Idft(src: Mat, dst: Mat, flags: c_int, nonzeroRows: c_int) void;
//*    pub extern fn Mat_InRange(src: Mat, lowerb: Mat, upperb: Mat, dst: Mat) void;
//*    pub extern fn Mat_InRangeWithScalar(src: Mat, lowerb: Scalar, upperb: Scalar, dst: Mat) void;
//*    pub extern fn Mat_InsertChannel(src: Mat, dst: Mat, coi: c_int) void;
//*    pub extern fn Mat_Invert(src: Mat, dst: Mat, flags: c_int) f64;
//*    pub extern fn KMeans(data: Mat, k: c_int, bestLabels: Mat, criteria: TermCriteria, attempts: c_int, flags: c_int, centers: Mat) f64;
//*    pub extern fn KMeansPoints(pts: PointVector, k: c_int, bestLabels: Mat, criteria: TermCriteria, attempts: c_int, flags: c_int, centers: Mat) f64;
//*    pub extern fn Mat_Log(src: Mat, dst: Mat) void;
//*    pub extern fn Mat_Magnitude(x: Mat, y: Mat, magnitude: Mat) void;
//*    pub extern fn Mat_Max(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_MeanStdDev(src: Mat, dstMean: Mat, dstStdDev: Mat) void;
//*    pub extern fn Mat_Merge(mats: struct_Mats, dst: Mat) void;
//*    pub extern fn Mat_Min(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_MinMaxIdx(m: Mat, minVal: [*c]f64, maxVal: [*c]f64, minIdx: [*c]c_int, maxIdx: [*c]c_int) void;
//*    pub extern fn Mat_MinMaxLoc(m: Mat, minVal: [*c]f64, maxVal: [*c]f64, minLoc: [*c]Point, maxLoc: [*c]Point) void;
//*    pub extern fn Mat_MixChannels(src: struct_Mats, dst: struct_Mats, fromTo: struct_IntVector) void;
//*    pub extern fn Mat_MulSpectrums(a: Mat, b: Mat, c: Mat, flags: c_int) void;
//*    pub extern fn Mat_Multiply(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_MultiplyWithParams(src1: Mat, src2: Mat, dst: Mat, scale: f64, dtype: c_int) void;
//*    pub extern fn Mat_Subtract(src1: Mat, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_Normalize(src: Mat, dst: Mat, alpha: f64, beta: f64, typ: c_int) void;
//*    pub extern fn Norm(src1: Mat, normType: c_int) f64;
//*    pub extern fn NormWithMats(src1: Mat, src2: Mat, normType: c_int) f64;
//*    pub extern fn Mat_PerspectiveTransform(src: Mat, dst: Mat, tm: Mat) void;
//*    pub extern fn Mat_Solve(src1: Mat, src2: Mat, dst: Mat, flags: c_int) bool;
//*    pub extern fn Mat_SolveCubic(coeffs: Mat, roots: Mat) c_int;
//*    pub extern fn Mat_SolvePoly(coeffs: Mat, roots: Mat, maxIters: c_int) f64;
//*    pub extern fn Mat_Reduce(src: Mat, dst: Mat, dim: c_int, rType: c_int, dType: c_int) void;
//*    pub extern fn Mat_Repeat(src: Mat, nY: c_int, nX: c_int, dst: Mat) void;
//*    pub extern fn Mat_ScaleAdd(src1: Mat, alpha: f64, src2: Mat, dst: Mat) void;
//*    pub extern fn Mat_SetIdentity(src: Mat, scalar: f64) void;
//*    pub extern fn Mat_Sort(src: Mat, dst: Mat, flags: c_int) void;
//*    pub extern fn Mat_SortIdx(src: Mat, dst: Mat, flags: c_int) void;
//*    pub extern fn Mat_Split(src: Mat, mats: [*c]struct_Mats) void;
//*    pub extern fn Mat_Trace(src: Mat) Scalar;
//*    pub extern fn Mat_Transform(src: Mat, dst: Mat, tm: Mat) void;
//*    pub extern fn Mat_Transpose(src: Mat, dst: Mat) void;
//*    pub extern fn Mat_PolarToCart(magnitude: Mat, degree: Mat, x: Mat, y: Mat, angleInDegrees: bool) void;
//*    pub extern fn Mat_Pow(src: Mat, power: f64, dst: Mat) void;
//*    pub extern fn Mat_Phase(x: Mat, y: Mat, angle: Mat, angleInDegrees: bool) void;
//*    pub extern fn Mat_Sum(src1: Mat) Scalar;
//*    pub extern fn Mat_rowRange(m: Mat, startrow: c_int, endrow: c_int) Mat;
//*    pub extern fn Mat_colRange(m: Mat, startrow: c_int, endrow: c_int) Mat;
