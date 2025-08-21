#pragma once

#include <shared_mutex>
#include <cstddef>
#include <list>
#include <memory>
#include <unordered_map>
#include <mutex>



template <typename TKey, typename TValue>
class Cache
{
public:
    using Pointer = std::shared_ptr<Cache<TKey, TValue>>;
    virtual void setCapacity(std::size_t newCapacity) = 0;
    virtual std::size_t capacity() const = 0;
    virtual std::size_t size() const = 0;
    virtual TValue get(const TKey& k) = 0;
    virtual TValue* getPointer(const TKey& k) = 0;
    virtual void put(const TKey& k, const TValue& v) = 0;
    virtual bool contains(const TKey& k) = 0;
    virtual void purge() = 0;

protected:
    Cache() = default;
    explicit Cache(std::size_t capacity) : capacity_{capacity} {}
    std::size_t capacity_{200};
};


template <typename TKey, typename TValue>
class LRUCache final : public Cache<TKey, TValue>
{
public:
    using BaseClass = Cache<TKey, TValue>;
    using BaseClass::capacity_;
    using TPair = typename std::pair<TKey, TValue>;
    using TListIterator = typename std::list<TPair>::iterator;
    using Pointer = std::shared_ptr<LRUCache<TKey, TValue>>;
    LRUCache() : BaseClass() {}
    explicit LRUCache(std::size_t capacity) : BaseClass(capacity) {}
    static auto New() -> Pointer
    {
        return std::make_shared<LRUCache<TKey, TValue>>();
    }
    static auto New(std::size_t capacity) -> Pointer
    {
        return std::make_shared<LRUCache<TKey, TValue>>(capacity);
    }
    void setCapacity(std::size_t capacity) override
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        if (capacity <= 0) {
            throw std::invalid_argument(
                "Cannot create cache with capacity <= 0");
        }
        capacity_ = capacity;

        // Cleanup elements that exceed the capacity
        while (lookup_.size() > capacity_) {
            auto last = std::end(items_);
            last--;
            lookup_.erase(last->first);
            items_.pop_back();
        }
    }
    auto capacity() const -> std::size_t override { return capacity_; }
    auto size() const -> std::size_t override { return lookup_.size(); }
    auto get(const TKey& k) -> TValue override
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        auto lookupIter = lookup_.find(k);
        if (lookupIter == std::end(lookup_)) {
            throw std::invalid_argument("Key not in cache");
        } else {
            items_.splice(std::begin(items_), items_, lookupIter->second);
            return lookupIter->second->second;
        }
    }
    auto getPointer(const TKey& k) -> TValue* override
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        auto lookupIter = lookup_.find(k);
        if (lookupIter == std::end(lookup_)) {
            throw std::invalid_argument("Key not in cache");
        } else {
            items_.splice(std::begin(items_), items_, lookupIter->second);
            return &lookupIter->second->second;
        }
    }
    void put(const TKey& k, const TValue& v) override
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        // If already in cache, need to refresh it
        auto lookupIter = lookup_.find(k);
        if (lookupIter != std::end(lookup_)) {
            items_.erase(lookupIter->second);
            lookup_.erase(lookupIter);
        }

        items_.push_front(TPair(k, v));
        lookup_[k] = std::begin(items_);

        if (lookup_.size() > capacity_) {
            auto last = std::end(items_);
            last--;
            lookup_.erase(last->first);
            items_.pop_back();
        }
    }
    auto contains(const TKey& k) -> bool override
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return lookup_.find(k) != std::end(lookup_);
    }
void purge() override
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        lookup_.clear();
        items_.clear();
    }
private:
    std::list<TPair> items_;
    std::unordered_map<TKey, TListIterator> lookup_;
    mutable std::shared_mutex cache_mutex_;
};

