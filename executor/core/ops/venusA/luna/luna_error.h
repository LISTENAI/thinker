#ifndef __LUNA_ERROR_H__
#define __LUNA_ERROR_H__

#define FAIL        (-1)
#define SUCCESS     (0)

#ifndef LUNA_ERR_TYPE_TYPEDEF
#define LUNA_ERR_TYPE_TYPEDEF

typedef enum
{
	LUNA_DATA_NOT_ALIGNED 		= 100000,
	LUNA_CACHE_SIZE_OVERFLOW	= 100001,
	LUNA_EXECUTE_FAIL			= 100002,

	LUNA_CMD_BUFFER_NEARLY_OVERFLOW 	= 100003, //internal CMD buffer is a little too small to hold cmds generated. a little dangerous, but not death kind.
	LUNA_CMD_BUFFER_OVERFLOW 		= 100004, //internal CMD buffer is too small to hold cmds generated. death kind.
	LUNA_SAVE_USER_BUFFER_OVERFLOW 	= 100005, //user buffer is too small to save internal CMDS generated.

}luna_err_type;

#endif //LUNA_ERR_TYPE_TYPEDEF

#endif /* __LUNA_ERROR_H__ */
