#ifndef MAPPED_FILE_HPP
#define MAPPED_FILE_HPP

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include <string>
#include <stdexcept>
#include <future>
#include <chrono>
#include <iostream>
#include <cassert>

#include "namespace.hpp"
LIBMARV_NAMESPACE_BEGIN

class MappedFileException : public std::exception{
    std::string message;
public:
    MappedFileException() : MappedFileException("MMapException"){}
    MappedFileException(const std::string& msg) : message(msg){}

    const char* what() const noexcept override{
        return message.c_str();
    }
};

struct MappedFile{
    struct Options{
        bool readaccess = true;
        bool writeaccess = false;
        bool prefault = false;
    };

#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap = NULL;
#else
    int fd;
#endif
    size_t filesize;
    void* rawMmapPtr;
    std::string filename;
    Options options;

    MappedFile(const std::string& filename_, MappedFile::Options options_)
        :   filesize(getFileSizeInBytes(filename_)),
            filename(filename_),
            options(options_){
#ifdef _WIN32
        DWORD access = 0;
        DWORD share = FILE_SHARE_READ;
        DWORD mapAccess = 0;
        DWORD viewAccess = 0;
        if(options.readaccess && options.writeaccess){
            access = GENERIC_READ | GENERIC_WRITE;
            mapAccess = PAGE_READWRITE;
            viewAccess = FILE_MAP_READ | FILE_MAP_WRITE;
        }else if(options.readaccess){
            access = GENERIC_READ;
            mapAccess = PAGE_READONLY;
            viewAccess = FILE_MAP_READ;
        }else{
            throw MappedFileException("Invalid options for MappedFile");
        }
        hFile = CreateFileA(filename.c_str(), access, share, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if(hFile == INVALID_HANDLE_VALUE){
            throw MappedFileException("Could not open file " + filename);
        }
        LARGE_INTEGER sz;
        GetFileSizeEx(hFile, &sz);
        filesize = static_cast<size_t>(sz.QuadPart);
        hMap = CreateFileMappingA(hFile, NULL, mapAccess, 0, 0, NULL);
        if(hMap == NULL){
            CloseHandle(hFile);
            throw MappedFileException("Could not create file mapping for " + filename);
        }
        rawMmapPtr = MapViewOfFile(hMap, viewAccess, 0, 0, 0);
        if(rawMmapPtr == NULL){
            CloseHandle(hMap);
            CloseHandle(hFile);
            throw MappedFileException("Could not map file " + filename);
        }
#else
        int openflags = 0;
        if(options.readaccess && options.writeaccess){
            openflags = O_RDWR;
        }else if(options.readaccess && !options.writeaccess){
            openflags = O_RDONLY;
        }else if(!options.readaccess && options.writeaccess){
            openflags = O_WRONLY;
        }else{
            throw MappedFileException("Invalid options for MappedFile");
        }

        fd = open(filename.c_str(), openflags);
        if(fd == -1){
            perror("open");
            throw MappedFileException("Could not open file " + filename);
        }

        int mmapprot = 0;
        if(options.readaccess){
            mmapprot |= PROT_READ;
        }
        if(options.writeaccess){
            mmapprot |= PROT_WRITE;
        }

        int mmapflags = MAP_PRIVATE;
        if(options.writeaccess){
            mmapflags = MAP_SHARED;
        }
        if(options.prefault){
            mmapflags |= MAP_POPULATE; //load the file into memory immediately
        }

        rawMmapPtr = mmap(nullptr, filesize, mmapprot, mmapflags, fd, 0);
        if(rawMmapPtr == MAP_FAILED){
            close(fd);
            throw MappedFileException("Could not map file " + filename);
        }
#endif
    }

    ~MappedFile(){
#ifdef _WIN32
        if(rawMmapPtr) UnmapViewOfFile(rawMmapPtr);
        if(hMap) CloseHandle(hMap);
        if(hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        munmap(rawMmapPtr, filesize);
        close(fd);
#endif
    }

    char* data() noexcept{
        return (char*)rawMmapPtr;
    }
    const char* data() const noexcept{
        return (const char*)rawMmapPtr;
    }
    size_t size() const noexcept{
        return filesize;
    }

    template<class T>
    size_t numElements() const noexcept{
        return size() / sizeof(T);
    }
private:
    size_t getFileSizeInBytes(const std::string& filename){
#ifdef _WIN32
        HANDLE h = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if(h == INVALID_HANDLE_VALUE){
            throw MappedFileException("Could not determine file size of file " + filename);
        }
        LARGE_INTEGER sz;
        if(!GetFileSizeEx(h, &sz)){
            CloseHandle(h);
            throw MappedFileException("Could not determine file size of file " + filename);
        }
        CloseHandle(h);
        return static_cast<size_t>(sz.QuadPart);
#else
        struct stat stat_buf;
        int rc = stat(filename.c_str(), &stat_buf);
        if(rc == 0){
            return stat_buf.st_size;
        }else{
            throw MappedFileException("Could not determine file size of file " + filename);
        }
#endif
    }
};


LIBMARV_NAMESPACE_END

#endif
