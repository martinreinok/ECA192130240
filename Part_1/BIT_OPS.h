/*
Bit Operations
*/
// Return data.bit value
#define get_bit(data, bit) ( (data>>bit) & 0b1)

// Set data.bit to 1
#define set_bit_high(data, bit) data |= (0b1 << bit)

// Set data.bit to 0
#define set_bit_low(data, bit)  data &= ~(0b1 << bit)

// Return bit range
#define get_bit_range(data, start_bit, end_bit) ((data >> start_bit) & ((0b1 << (end_bit - start_bit)) - 0b1))
